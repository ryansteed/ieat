from ieat.models import SENTExtractor, OpenAIExtractor, LogitExtractor
from weat.test import Test

import logging

import os
import glob
from collections import namedtuple
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
progress_level = 25
logging.addLevelName(progress_level, "PROGRESS")
def progress(self, message, *args, **kws):
	self._log(progress_level, message, args, **kws)
logging.Logger.progress = progress


def test(
	X, Y, A, B, # content
	model_type, model_size, models_dir, clusters_dir, n_px, # model details
	file_types=(".jpg", ".jpeg", ".png", ".webp"),
	from_cache=True,
	verbose=False,
	gpu=False,
	batch_size=20,
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
	extractor = _load_model(model_type, model_size, models_dir, clusters_dir, n_px, from_cache)
	assert extractor is not None, f"Model type '{model_type}' not found."

	for d in input_dirs:
		logger.progress(f"Extracting images from {d}")
		embeddings.append(extractor.extract_dir(
			d, file_types, 
			visualize=verbose, 
			gpu=gpu,
			batch_size=batch_size
		))
	assert len(embeddings) is not None, "Embeddings could not be extracted."
	assert len(embeddings) == len(input_dirs), "Not all embeddings could not be extracted."

	# run the test
	logger.info("Running test")
	test = Test(*embeddings, names=[os.path.basename(d) for d in input_dirs])
	return test.run(**test_params)


def test_all(
		model_types, model_size, models_dir, clusters_dir, n_px,  # model details
		tests=None,
		**test_params
	):
	"""
	Produce a table of model_type x test results.
	:param model_types: List of models to test.
	:param model_size: iGPT model size - e.g. 's', 'm', or 'l'
	:param models_dir: directory of iGPT model checkpoints
	:param clusters_dir: directory of iGPT color cluster files
	:param n_px: number of pixels in the input images
	:param test_params: Extra params for the `test` method
	:param tests: Optional list of tests to run, by name - see `tests_all`
	:return: the test effect size and p-value
	:return:
	"""
	TestData = namedtuple('TestData', ['name', 'X', 'Y', 'A', 'B'])
	tests_all = [
		# Baseline
		TestData(
			'Insect-Flower', 'insect-flower/flower', 'insect-flower/insect', 'valence/pleasant', 'valence/unpleasant'
		),
		# Picture-Picture IATS
		TestData('Weapon', 'weapon/white', 'weapon/black', 'weapon/tool', 'weapon/weapon'),
		TestData('Weapon (Modern)', 'weapon/white', 'weapon/black', 'weapon/tool-modern', 'weapon/weapon-modern'),
		TestData('Native', 'native/euro', 'native/native', 'native/us', 'native/world'),
		TestData('Asian', 'asian/european-american', 'asian/asian-american', 'asian/american', 'asian/foreign'),
		# Valence IATs
		TestData('Weight', 'weight/thin', 'weight/fat', 'valence/pleasant', 'valence/unpleasant'),
		TestData('Skin-Tone', 'skin-tone/light', 'skin-tone/dark', 'valence/pleasant', 'valence/unpleasant'),
		TestData('Disability', 'disabled/disabled', 'disabled/abled', 'valence/pleasant', 'valence/unpleasant'),
		TestData(
			'President - Kennedy vs. Trump',
			'presidents/kennedy', 'presidents/trump', 'valence/pleasant', 'valence/unpleasant'
		),
		TestData(
			'President - B. Clinton vs. Trump',
			'presidents/clinton', 'presidents/trump', 'valence/pleasant', 'valence/unpleasant'
		),
		TestData(
			'President - Bush vs. Trump',
			'presidents/bush', 'presidents/trump', 'valence/pleasant', 'valence/unpleasant'
		),
		TestData(
			'President - Lincoln vs. Trump',
			'presidents/lincoln', 'presidents/trump', 'valence/pleasant', 'valence/unpleasant'
		),
		TestData('Religion', 'religion/christianity', 'religion/judaism', 'valence/pleasant', 'valence/unpleasant'),
		TestData('Sexuality', 'sexuality/gay', 'sexuality/straight', 'valence/pleasant', 'valence/unpleasant'),
		TestData('Race', 'race/european-american', 'race/african-american', 'valence/pleasant', 'valence/unpleasant'),
		TestData(
			'Arab-Muslim',
			'arab-muslim/other-people', 'arab-muslim/arab-muslim', 'valence/pleasant', 'valence/unpleasant'
		),
		TestData('Age', 'age/young', 'age/old', 'valence/pleasant', 'valence/unpleasant'),
		# Stereotype IATS
		TestData('Gender-Science', 'gender/science', 'gender/liberal-arts', 'gender/male', 'gender/female'),
		TestData('Gender-Career', 'gender/career', 'gender/family', 'gender/male', 'gender/female')
	]

	logger.setLevel(progress_level)

	results = {}
	to_test = tests_all if tests is None else (t for t in tests_all if t.name in tests)
	for test_data in to_test:
		# logger.progress(f"Running {test_data.name}")
		print(f"## {test_data.name} ##")
		for model_type in model_types:
			print(f"# {model_type} #")
			categories = [
				os.path.join('data/experiments', cat) for cat in (test_data.X, test_data.Y, test_data.A, test_data.B)
			]
			effect, p = test(
				*categories,
				model_type=model_type,
				model_size=model_size,
				models_dir=models_dir,
				clusters_dir=clusters_dir,
				n_px=n_px,
				**test_params
			)
			# pull the sample sizes for X and A
			n_target, n_attr = (len(glob.glob1(categories[c], "*")) for c in [0, 2])
			results[(test_data.name, model_type)] = (*categories, effect, p, n_target, n_attr)

	return results


def _load_model(model_type, model_size, models_dir, clusters_dir, n_px, from_cache):
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
	return models.get(model_type)


if __name__ == "__main__":
	# some default settings
	model_size = "l"
	models_dir = "models"
	color_clusters_dir = "clusters"
	n_px = 32

	print(test_all(
	    model_types=["logit", "openai"],
	    model_size=model_size,
	    models_dir=models_dir,
	    clusters_dir=color_clusters_dir,
	    gpu=True,
	    n_px=n_px
	))