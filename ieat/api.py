from ieat.models import SENTExtractor, OpenAIExtractor, LogitExtractor, SimCLRExtractor
from weat.test import Test

import logging

import os
import glob
from collections import namedtuple

logger = logging.getLogger()
logger.setLevel(logging.INFO)
progress_level = 25
logging.addLevelName(progress_level, "PROGRESS")
def progress(self, message, *args, **kws):
	self._log(progress_level, message, args, **kws)
logging.Logger.progress = progress


def test(
	X, Y, A, B,  # content
	model_type: str,
	model_params: list,  # model parameters
	file_types=(".jpg", ".jpeg", ".png", ".webp"),
	from_cache=True,
	verbose=False,
	gpu=False,
	batch_size=20,
	model=None,
	**test_params
):
	"""
	:param X: a directory of target images
	:param Y: a directory of target images
	:param A: a directory of attribute images
	:param B: a directory of attribute images
	:param model_type: key name of model
	:param model_params: Model-specific initialization parameters
	:param file_types: acceptable image file types
	:param from_cache: whether to use cached embeddings at the location `embedding_path`
	:param verbose: whether to print out images, other detailed logging info
	:return: the test effect size and p-value
	"""

	input_dirs = [X, Y, A, B]
	for d in input_dirs: assert os.path.exists(d), "%s is not a valid path." % d

	# get the embeddings
	embeddings = []
	extractor = model if model is not None else _load_model(
		model_type, *model_params, from_cache=from_cache
	)
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
		model_types: dict,
		tests: list = None,
		from_cache=True,
		**test_params
	):
	"""
	Produce a table of model_type x test results.
	:param model_types: mapping of model type keyword to parameters for that model
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
		TestData('Gender-Career', 'gender/career', 'gender/family', 'gender/male', 'gender/female'),
		# Intersectional IATs
		# - Gender Stereotypes
		TestData(
			'Intersectional-Gender-Science-MF', 'gender/science', 'gender/liberal-arts', 'intersectional/male',
			'intersectional/female'
		),
		TestData(
			'Intersectional-Gender-Science-WMBM', 'gender/science', 'gender/liberal-arts', 'intersectional/white-male',
			'intersectional/black-male'
		),
		TestData(
			'Intersectional-Gender-Science-WMBF', 'gender/science', 'gender/liberal-arts', 'intersectional/white-male',
			'intersectional/black-female'
		),
		TestData(
			'Intersectional-Gender-Science-WMWF', 'gender/science', 'gender/liberal-arts', 'intersectional/white-female',
			'intersectional/white-male'
		),
		TestData(
			'Intersectional-Gender-Career-MF', 'gender/career', 'gender/family', 'intersectional/male',
			'intersectional/female'
		),
		TestData(
			'Intersectional-Gender-Career-WMBM', 'gender/career', 'gender/family', 'intersectional/black-male',
			'intersectional/white-male'
		),
		TestData(
			'Intersectional-Gender-Career-WMBF', 'gender/career', 'gender/family', 'intersectional/white-male',
			'intersectional/black-female'
		),
		TestData(
			'Intersectional-Gender-Career-WMWF', 'gender/career', 'gender/family', 'intersectional/white-male',
			'intersectional/white-female'
		),
		# - Valence
		TestData(
			'Intersectional-Valence-BW', 'intersectional/white', 'intersectional/black', 'valence/pleasant',
			'valence/unpleasant'
		),
		TestData(
			'Intersectional-Valence-WMBM', 'intersectional/white-male', 'intersectional/black-male', 'valence/pleasant',
			'valence/unpleasant'
		),
		TestData(
			'Intersectional-Valence-WMBF', 'intersectional/white-male', 'intersectional/black-female', 'valence/pleasant',
			'valence/unpleasant'
		),
		TestData(
			'Intersectional-Valence-WMWF', 'intersectional/white-female', 'intersectional/white-male', 'valence/pleasant',
			'valence/unpleasant'
		),
		TestData(
			'Intersectional-Valence-WFBM', 'intersectional/white-female', 'intersectional/black-male', 'valence/pleasant',
			'valence/unpleasant'
		)
	]

	logger.setLevel(progress_level)

	results = {}
	to_test = tests_all if tests is None else (t for t in tests_all if t.name in tests)
	for model_type, model_params in model_types.items():
		print(f"# {model_type} #")
		extractor = _load_model(
			model_type, *model_params, from_cache=from_cache
		)
		for test_data in to_test:
			print(f"## {test_data.name} ##")
			categories = [
				os.path.join('data/experiments', cat) for cat in (test_data.X, test_data.Y, test_data.A, test_data.B)
			]
			effect, p = test(
				*categories,
				model_type,
				model_params,
				model=extractor,
				**test_params
			)
			# pull the sample sizes for X and A
			n_target, n_attr = (len(glob.glob1(categories[c], "*")) for c in [0, 2])
			results[(test_data.name, model_type)] = (*categories, effect, p, n_target, n_attr)

	return results


def _load_model(model_type, *model_params, **model_kwargs):
	if model_type == "igpt-logit":
		return LogitExtractor(
			model_type,
			*model_params,
			**model_kwargs
		)
	elif model_type == "sent":
		return SENTExtractor(
			model_type,
			*model_params,
			**model_kwargs
		)
	elif model_type == "igpt":
		return OpenAIExtractor(
			model_type,
			*model_params,
			**model_kwargs
		)
	elif model_type == "simclr":
		return SimCLRExtractor(
			model_type,
			*model_params,
			**model_kwargs
		)
	raise ValueError(f"Invalid model type {model_type}.")


if __name__ == "__main__":
	# some default settings
	model_size = "l"
	models_dir = "models"
	color_clusters_dir = "clusters"
	n_px = 32
	depth = 50
	width = 1
	sk = 0

	print(test_all(
		model_types={
			"igpt-logit": (
			 	model_size,
			 	models_dir,
			 	color_clusters_dir,
			 	n_px
			 ),
			"igpt": (
				model_size,
			 	models_dir,
			 	color_clusters_dir,
			 	n_px
			)
			#"simclr": (
			#	depth,
			#	width,
			#	sk
			#)
		},
		gpu=False,
		from_cache=True
	))
