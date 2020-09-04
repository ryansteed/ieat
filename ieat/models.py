## Code adapted from https://colab.research.google.com/github/apeguero1/image-gpt/blob/master/Transformers_Image_GPT.ipynb 
## - thanks to the author

from ieat.utils import resize, normalize_img, color_quantize_np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import transformers
from transformers.modeling_gpt2 import GPT2Model,GPT2LMHeadModel

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging
logger = logging.getLogger()


class EmbeddingExtractor:
	MODELS={"l":(1536,16,48),"m":(1024,8,36),"s":(512,8,24)} 

	def __init__(self, model_size, models_dir, color_clusters_dir, n_px):
		"""
		:param model_dir: path to a directory containing the model with the parameters specified
		"""
		self.n_px = n_px

		color_clusters_file = "%s/kmeans_centers.npy"%(color_clusters_dir)
		self.clusters = np.load(color_clusters_file) #get color clusters
		
		n_embd, n_head, n_layer = EmbeddingExtractor.MODELS[model_size] #set model hyperparameters

		self.vocab_size = len(self.clusters) + 1 #add one for start of sentence token
		config = transformers.GPT2Config(
			vocab_size=self.vocab_size,
			n_ctx=n_px*n_px,
			n_positions=n_px*n_px,
			n_embd=n_embd,
			n_layer=n_layer,
			n_head=n_head
		)
		model_path = "%s/%s/model.ckpt-1000000.index"%(models_dir, model_size)
		self.model = ImageGPT2LMHeadModel.from_pretrained(model_path,from_tf=True,config=config)

	def extract(self, image_paths, output_path=None, gpu=False, **process_kwargs):
		samples = self.process_samples(image_paths, **process_kwargs)
		with torch.no_grad(): # saves some memory
			# initialize with SOS token
			context = np.concatenate( 
				(
					np.full( (len(image_paths), 1), self.vocab_size - 1 ),
					samples.reshape(-1, self.n_px*self.n_px),
				), axis=1
			)
			# DEBUG THIS LATER
			# must drop the last pixel to make room for the SOS
			context = torch.tensor(context[:,:-1]) if not gpu else torch.tensor(context[:,:-1]).cuda()
			enc, _ = self.model(context)

			enc_last = enc[:, -1, :].numpy() if not gpu else enc[:, -1, :].cpu().numpy()  # extract the rep of the last input, as in sent-bias

			df = pd.DataFrame(enc_last)
			df["img"] = [os.path.basename(path) for path in image_paths]
			df["category"] = [os.path.basename(os.path.dirname(path)) for path in image_paths]
			
			if output_path is not None:
				# add the image names to the CSV file
				df.to_csv(output_path)

			return df.set_index("img")

	def process_samples(self, image_paths, visualize=False):
		for path in image_paths: assert os.path.exists(path), "ERR: %s is not a valid path." % path
		# print("Num paths: %s" % len(image_paths))
		x = resize(self.n_px, image_paths)
		# print("X shape: ", x.shape)
		x_norm = normalize_img(x) #normalize pixels values to -1 to +1
		samples = color_quantize_np(x_norm, self.clusters).reshape(x_norm.shape[:-1]) #map pixels to closest color cluster
		
		if visualize:
			samples_img = [
				np.reshape(
					np.rint(127.5 * (self.clusters[s] + 1.0)), [self.n_px, self.n_px, 3]
				).astype(np.uint8) for s in samples
			] # convert color clusters back to pixels
			f, axes = plt.subplots(1,len(image_paths),dpi=300)
			for img, ax in zip(samples_img, axes):
				ax.axis('off')
				ax.imshow(img)
		# print("Shape of samples: ", samples.shape)
		return samples


class ln_mod(nn.Module):
	def __init__(self, nx,eps=1e-5):
		super().__init__()
		self.eps = eps
		self.weight = Parameter(torch.Tensor(nx))

	def forward(self,x): #input is not mean centered
		return x / torch.sqrt( torch.std(x, axis=-1, unbiased=False, keepdim=True)**2 + self.eps ) * self.weight.data[...,:]        


def load_tf_weights_in_image_gpt2(model, config, gpt2_checkpoint_path):
	""" 
	Load tf checkpoints in a pytorch model
	"""
	try:
		import re
		import tensorflow as tf
	except ImportError:
		logger.error(
			"Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
			"https://www.tensorflow.org/install/ for installation instructions."
		)
		raise
	tf_path = os.path.abspath(gpt2_checkpoint_path)
	logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
	# Load weights from TF model
	init_vars = tf.train.list_variables(tf_path)
	names = []
	arrays = []

	for name, shape in init_vars:
		logger.info("Loading TF weight {} with shape {}".format(name, shape))
		array = tf.train.load_variable(tf_path, name)
		names.append(name)
		arrays.append(array.squeeze())

	for name, array in zip(names, arrays):
		name = name[6:]  # skip "model/"
		name = name.split("/")

		# adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
		# which are not required for using pretrained model
		if any(
			n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
			for n in name
		) or name[-1] in ['_step']:
			logger.info("Skipping {}".format("/".join(name)))
			continue
		
		pointer = model
		if name[-1] not in ["wtet"]:
		  pointer = getattr(pointer, "transformer")
		
		for m_name in name:
			if re.fullmatch(r"[A-Za-z]+\d+", m_name):
				scope_names = re.split(r"(\d+)", m_name)
			else:
				scope_names = [m_name]

			if scope_names[0] == "w" or scope_names[0] == "g":
				pointer = getattr(pointer, "weight")
			elif scope_names[0] == "b":
				pointer = getattr(pointer, "bias")
			elif scope_names[0] == "wpe" or scope_names[0] == "wte":
				pointer = getattr(pointer, scope_names[0])
				pointer = getattr(pointer, "weight")
			elif scope_names[0] in ['q_proj','k_proj','v_proj']:
				pointer = getattr(pointer, 'c_attn')
				pointer = getattr(pointer, 'weight')
			elif len(name) ==3 and name[1]=="attn" and scope_names[0]=="c_proj":
				pointer = getattr(pointer, scope_names[0])
				pointer = getattr(pointer, 'weight')
			elif scope_names[0]=="wtet":
				pointer = getattr(pointer, "lm_head")
				pointer = getattr(pointer, 'weight')
			elif scope_names[0]=="sos":
				pointer = getattr(pointer,"wte")
				pointer = getattr(pointer, 'weight')
			else:
				pointer = getattr(pointer, scope_names[0])
			if len(scope_names) >= 2:
				num = int(scope_names[1])
				pointer = pointer[num]

		if len(name) > 1 and name[1]=="attn" or name[-1]=="wtet" or name[-1]=="sos" or name[-1]=="wte":
		   pass #array is used to initialize only part of the pointer so sizes won't match
		else:
		  try:
			  assert pointer.shape == array.shape
		  except AssertionError as e:
			  e.args += (pointer.shape, array.shape)
			  raise
		  
		logger.info("Initialize PyTorch weight {}".format(name))

		if name[-1]=="q_proj":
		  pointer.data[:,:config.n_embd] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
		elif name[-1]=="k_proj":
		  pointer.data[:,config.n_embd:2*config.n_embd] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
		elif name[-1]=="v_proj":
		  pointer.data[:,2*config.n_embd:] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
		elif (len(name) ==3 and name[1]=="attn" and name[2]=="c_proj" ):
		  pointer.data = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) )
		elif name[-1]=="wtet":
		  pointer.data = torch.from_numpy(array)
		elif name[-1]=="wte":
		  pointer.data[:config.vocab_size-1,:] = torch.from_numpy(array)
		elif name[-1]=="sos":
		  pointer.data[-1] = torch.from_numpy(array)
		else:
		  pointer.data = torch.from_numpy(array)

	return model


def replace_ln(m, name, config):
	for attr_str in dir(m):
		target_attr = getattr(m, attr_str)
		if type(target_attr) == torch.nn.LayerNorm:
			setattr(m, attr_str, ln_mod(config.n_embd,config.layer_norm_epsilon))

	for n, ch in m.named_children():
		replace_ln(ch, n,config) 


class ImageGPT2LMHeadModel(GPT2LMHeadModel):
	load_tf_weights = load_tf_weights_in_image_gpt2

	def __init__(self, config):
		super().__init__(config)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
		replace_ln(self, "net",config) #replace layer normalization
		for n in range(config.n_layer):
			self.transformer.h[n].mlp.act = ImageGPT2LMHeadModel.gelu2 #replace activation 

	def tie_weights(self): #image-gpt doesn't tie output and input embeddings
		pass 

	@staticmethod
	def gelu2(x):
		return x * torch.sigmoid(1.702 * x)
