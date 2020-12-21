from ieat.models import SENTExtractor, OpenAIExtractor, LogitExtractor, SimCLRExtractor
from weat.test import Test
from ieat.utils import tests_all, TestData

import logging

import os
import glob

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
