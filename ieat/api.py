from ieat.models import SENTExtractor, OpenAIExtractor, LogitExtractor
from weat.test import Test

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os
import pandas as pd


def test(
	X, Y, A, B, # content
	model_type, model_size, models_dir, clusters_dir, n_px, # model details
	file_types=[".jpg", ".jpeg", ".png", ".webp"],
	from_cache=True,
	verbose=True,
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
	:param from_cache: whether to use cached embeddings at the location `embedding_path`
	:param verbose: whether to print out images, other detailed logging info
	:return: the test effect size and p-value
	"""
	
	input_dirs = [X, Y, A, B]
	for d in input_dirs: assert os.path.exists(d), "%s is not a valid path." % d

	# get the embeddings
	embeddings = []
	models = {
		"logit": LogitExtractor(
			model_type,
			model_size=model_size,
			models_dir=models_dir,
			color_clusters_dir=clusters_dir,
			n_px=n_px,
			from_cache=from_cache
		),
		"sent": SENTExtractor(
			model_type,
			model_size=model_size,
			models_dir=models_dir,
			color_clusters_dir=clusters_dir,
			n_px=n_px,
			from_cache=from_cache
		),
		"openai": OpenAIExtractor(
			model_type,
			model_size=model_size,
			models_dir=models_dir,
			color_clusters_dir=clusters_dir,
			n_px=n_px,
			from_cache=from_cache
		)
	}
	extractor = models.get(model_type)
	assert extractor is not None, f"Model type '{model_type}' not found."

	for d in input_dirs:
		embeddings.append(extractor.extract_dir(d, file_types, visualize=verbose))
	assert len(embeddings) is not None, "Embeddings could not be extracted."
	assert len(embeddings) == len(input_dirs), "Not all embeddings could not be extracted."

	# run the test
	logger.info("Running test")
	test = Test(*embeddings, names=[os.path.basename(d) for d in input_dirs])
	return test.run(**test_params)
