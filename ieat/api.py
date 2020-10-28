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
def _progress(self, message, *args, **kws):
	self._log(progress_level, message, args, **kws)
logging.Logger.progress = _progress


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
	Parameters
	----------
	X : str
		a directory of target images
	Y : str
		a directory of target images
	A : str
		a directory of attribute images
	B : str
		a directory of attribute images
	model_type : str
		key name of model
	model_params : dict
		Model-specific initialization parameters
	file_types : list[str]
		acceptable image file types
	from_cache : bool
		whether to use cached embeddings at the location `embedding_path`
	verbose : bool
		whether to print out images, other detailed logging info
	gpu : bool
		whether to use GPU (True) or CPU (False)
	batch_size : int
		batch size of processing - helps when you have limited memory
	model : str
		name of the model being tested - used for caching
	test_params : dict
		additional test params

	Returns
	-------
	d : float
		the test effect size
	p : float
		the p-value
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
	Produces a table of model_type x test results.
	Parameters
	----------
	model_types : dict[str, dict]
		mapping of model type keyword to parameters for that model
	tests : list[str]
		Optional list of tests to run, by name - see source code for the keys
	from_cache : bool
		Whether to use the cache
	test_params : dict
		additional test params

	Returns
	-------
	results : dict[tuple, tuple]
		results of the tests, mapped by model and test -> categories used, effect size, p value, target sample size,
		and attribute sample size
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
		TestData('Gender-Science', 'gender/male', 'gender/female', 'gender/science', 'gender/liberal-arts'),
		TestData('Gender-Career', 'gender/male', 'gender/female', 'gender/career', 'gender/family'),
		# Intersectional IATs
		# - Gender Stereotypes
		TestData(
			'Intersectional-Gender-Science-MF', 'intersectional/male',
			'intersectional/female', 'gender/science', 'gender/liberal-arts'
		),
		TestData(
			'Intersectional-Gender-Science-WMBM', 'intersectional/white-male',
			'intersectional/black-male', 'gender/science', 'gender/liberal-arts'
		),
		TestData(
			'Intersectional-Gender-Science-WMBF', 'intersectional/white-male',
			'intersectional/black-female', 'gender/science', 'gender/liberal-arts'
		),
		TestData(
			'Intersectional-Gender-Science-WMWF', 'intersectional/white-male',
			'intersectional/white-female', 'gender/science', 'gender/liberal-arts'
		),
		TestData(
			'Intersectional-Gender-Career-MF', 'intersectional/male',
			'intersectional/female', 'gender/career', 'gender/family'
		),
		TestData(
			'Intersectional-Gender-Career-WMBM', 'intersectional/black-male',
			'intersectional/white-male', 'gender/career', 'gender/family'
		),
		TestData(
			'Intersectional-Gender-Career-WMBF', 'intersectional/white-male',
			'intersectional/black-female', 'gender/career', 'gender/family'
		),
		TestData(
			'Intersectional-Gender-Career-WMWF', 'intersectional/white-male',
			'intersectional/white-female', 'gender/career', 'gender/family'
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
		),
		TestData(
			'Intersectional-Valence-BFBM', 'intersectional/black-female', 'intersectional/black-male', 'valence/pleasant',
			'valence/unpleasant'
		),
		TestData(
			'Intersectional-Valence-WFBF', 'intersectional/white-female', 'intersectional/black-female', 'valence/pleasant',
			'valence/unpleasant'
		),
		TestData(
			'Intersectional-Valence-FM', 'intersectional/female', 'intersectional/male', 'valence/pleasant',
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
