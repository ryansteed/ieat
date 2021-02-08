# Some code adapted from
# https://colab.research.google.com/github/apeguero1/image-gpt/blob/master/Transformers_Image_GPT.ipynb
# - thanks to the author

import cv2
import numpy as np
from collections import namedtuple

# numpy implementation of functions in image-gpt/src/utils which convert pixels of image to nearest color cluster.
# Resize original images to n_px by n_px


def normalize_img(img):
	return img/127.5 - 1


def squared_euclidean_distance_np(a,b):
	b = b.T
	a2 = np.sum(np.square(a),axis=1)
	b2 = np.sum(np.square(b),axis=0)
	ab = np.matmul(a,b)
	d = a2[:,None] - 2*ab + b2[None,:]
	return d


def color_quantize_np(x, clusters):
	x = x.reshape(-1, 3)
	d = squared_euclidean_distance_np(x, clusters)
	return np.argmin(d,axis=1)


def resize(n_px, image_paths, rotate_90=False):
	dim=(n_px,n_px)
	x = np.zeros((len(image_paths),n_px,n_px,3),dtype=np.uint8)

	for n,image_path in enumerate(image_paths):
		img_np = cv2.imread(image_path)   # reads an image in the BGR format
		img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)   # BGR -> RGB
		H,W,C = img_np.shape
		D = min(H,W)
		img_np = img_np[:D,:D,:C] #get square piece of image
		if (rotate_90):
			img_np = cv2.rotate(img_np, cv2.cv2.ROTATE_90_CLOCKWISE)
		x[n] = cv2.resize(img_np,dim, interpolation = cv2.INTER_AREA) #resize to n_px by n_px

	return x

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
        'Intersectional-Gender-Science-BMBF', 'intersectional/black-male',
		'intersectional/black-female', 'gender/science', 'gender/liberal-arts'  
    ),
    TestData(
        'Intersectional-Gender-Science-BMWF', 'intersectional/black-male',
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
	TestData(
		'Intersectional-Gender-Career-BMBF', 'intersectional/black-male',
		'intersectional/black-female', 'gender/career', 'gender/family'
	),
	TestData(
		'Intersectional-Gender-Career-BMWF', 'intersectional/black-male',
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
