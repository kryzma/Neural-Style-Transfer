import cv2 as cv
import numpy as np

import network as network
import torch
from torchvision import transforms

#============== Preparing Images ============#

def show_image(img):
	cv.imshow('img', img)
	cv.waitKey(0)

def load_image(img_path, target_shape=None):
	img = cv.imread(img_path)[:, :, ::-1] # reads in BGR format and reverses last dimension

	if target_shape is not None:
		img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

	img = img.astype(np.float32)
	return img

def prepare_img(img_path, target_shape, device, mean=[123.675, 116.28, 103.53], std=[1, 1, 1]):
	img = load_image(img_path, target_shape)

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
		])
	img = transform(img).to(device).unsqueeze(0)

	return img


def save_image(optimizing_img, cnt, config):

	path = config['output_dir'] + "img" + str(cnt) + ".jpg" 
	out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
	out_img = np.moveaxis(out_img, 0, 2) # move chanel to 3rd cord

	dump_img = np.copy(out_img)
	dump_img += np.array(IMAGE_MEAN).reshape((1,1,3))
	dump_img = np.clip(dump_img, 0, 255).astype('uint8')
	cv.imwrite(path, dump_img[:, :, ::-1]) # reverses last dimension, BGR -> RGB



#============== Other utilities ============#

def gram_matrix(x):
	(b, ch, h, w) = x.size()
	features = x.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t)
	gram /= ch * h * w
	return gram

def prepare_model(device):

	model = network.Vgg19()

	model.eval()

	return model.to(device), model.layers_names[model.content_feature_maps_index] , [model.layers_names[x] for x in model.style_feature_maps_indices]