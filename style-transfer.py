import utils as utils

import os

import torch
from torch.optim import Adam
from torch.autograd import Variable


def build_loss(neural_net, optimizing_img, target_content_representation, target_style_representation, content_feature_maps_name, style_feature_maps_names, config):

	current_set_of_feature_maps = neural_net(optimizing_img)

	current_content_representation = current_set_of_feature_maps[content_feature_maps_name].squeeze(axis=0) # ?
	content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

	style_loss = 0.0
	current_style_representation = [utils.gram_matrix(current_set_of_feature_maps[x]) for x in style_feature_maps_names]
	for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
		style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])

	style_loss /= len(target_content_representation)

	total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss

	return total_loss, content_loss, style_loss

def tuning_step(neural_net, optimizer, target_content_representation, target_style_representation, content_feature_maps_name, style_feature_maps_names, optimizing_img, config):

	total_loss, content_loss, style_loss = build_loss(neural_net, optimizing_img, target_content_representation, target_style_representation, content_feature_maps_name, style_feature_maps_names, config)

	total_loss.backward()
	optimizer.step()
	optimizer.zero_grad()
	return total_loss, content_loss, style_loss

def neural_style_transfer(config):
	content_image_path = os.path.join(config['content_images_dir'], config['content_image_name'])
	style_image_path = os.path.join(config['style_images_dir'], config['style_image_name'])
	init_image_path = os.path.join(config['init_images_dir'], config['init_image_name'])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	content_img = utils.prepare_img(content_image_path, (config['height'], config['height']), device)
	style_img = utils.prepare_img(style_image_path, (config['height'], config['height']), device)

	init_img = utils.prepare_img(init_image_path, (config['height'], config['height']), device)

	optimizing_img = Variable(init_img, requires_grad=True)

	neural_net, content_feature_maps_name, style_feature_maps_names = utils.prepare_model(device)

	content_img_set_of_feature_maps = neural_net(content_img)
	style_img_set_of_feature_maps = neural_net(style_img)

	target_content_representation = content_img_set_of_feature_maps[content_feature_maps_name]
	target_style_representation = [utils.gram_matrix(style_img_set_of_feature_maps[x]) for x in style_feature_maps_names ]


	optimizer = Adam((optimizing_img,), lr=config['learning_rate'])
	for cnt in range(config['num_of_iterations']):
		total_loss, content_loss, style_loss = tuning_step(neural_net, optimizer, target_content_representation, target_style_representation, content_feature_maps_name, style_feature_maps_names, optimizing_img, config)
		print(f"Iteration: {cnt:03}, tot loss={total_loss.item():9.5f}, content loss={config['content_weight']*content_loss.item():9.5f}, style loss={config['style_weight']*style_loss.item():9.5f}")
		if(cnt % 20 == 0):
			utils.save_image(optimizing_img, cnt, config)


if __name__ == "__main__":

	config = dict()
	config['output_dir'] = "C:/Users/Mantas/Desktop/StyleTransfer/images/results/"

	config['content_images_dir'] = "C:/Users/Mantas/Desktop/StyleTransfer/images/contents/"
	config['content_image_name'] = "noise.jpg"
	config['style_images_dir'] = "C:/Users/Mantas/Desktop/StyleTransfer/images/styles/"
	config['style_image_name'] = "tomato.jpg"
	config['init_images_dir'] = "C:/Users/Mantas/Desktop/StyleTransfer/images/contents/"
	config['init_image_name'] = "noise.jpg"

	config['height'] = 256

	config['num_of_iterations'] = 1000
	config['learning_rate'] = 1e2 # 1e1

	config['content_weight'] = 0 # 1e5
	config['style_weight'] = 2e5 # 1e5

	neural_style_transfer(config)







