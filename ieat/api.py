from ieat.models import EmbeddingExtractor
from weat.test import Test

import logging
logger = logging.getLogger()

import os
import pandas as pd


def test(
	X, Y, A, B, 
	model_size, models_dir, clusters_dir, n_px,
	file_types=[".jpg"],
	embedding_path=None,
	from_cache=True,
	**test_params
):
	"""
	:param X: a directory of target images
	:param Y: a directory of target images
	:param A: a directory of attribute images
	:param B: a directory of attribute images
	:param model_size: iGPT model size - e.g. 's', 'm', or 'l'
	:param models_dir: directory of iGPT model checkpoints
	:param clusters_dir: directory of iGPT color cluster files
	:param n_px: number of pixels in the input images
	:param file_types: acceptable image file types
	:param embedding_path: where to store the generated embeddings
	:param from_cache: whether to use cached embeddings at the location `embedding_path`
	:return: the test effect size and p-value
	"""
	
	input_dirs = [X, Y, A, B]
	for d in input_dirs: assert os.path.exists(d), "%s is not a valid path." % d
	if embedding_path is None: embedding_path = "embeddings/{}.csv".format(os.path.basename(os.path.dirname(X)))

	# get the embeddings
	encs = None
	if from_cache and os.path.exists(embedding_path):
		logger.info("Loading embeddings from file")
		encs = pd.read_csv(embedding_path, index_col=0).set_index("img")
	else:
		logger.info("Extracting embeddings")
		logger.setLevel(logging.WARN) 
		extractor = EmbeddingExtractor(
			model_size=model_size,
			models_dir=models_dir,
			color_clusters_dir=clusters_dir,
			n_px=n_px
		)
		image_paths = [
			os.path.join(d, f) for d in input_dirs for f in os.listdir(d) 
			if os.path.splitext(f)[1] in file_types
		]
		encs = extractor.extract(image_paths, output_path=embedding_path)
	assert encs is not None, "Embeddings could not be extracted."

	# run the test
	logger.info("Running test")
	logger.setLevel(logging.INFO) 
	embeddings = [
		encs.loc[[f for f in os.listdir(d) if os.path.splitext(f)[1] in file_types]] 
		for d in input_dirs
	]
	test = Test(*embeddings, names=[os.path.basename(d) for d in input_dirs])
	return test.run(**test_params)
