## Code adapted from https://colab.research.google.com/github/apeguero1/image-gpt/blob/master/Transformers_Image_GPT.ipynb 
## - thanks to the author

#Resize original images to n_px by n_px
import cv2
import numpy as np

#numpy implementation of functions in image-gpt/src/utils which convert pixels of image to nearest color cluster. 
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
