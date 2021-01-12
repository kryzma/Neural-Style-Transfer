import torch
import torchvision.models as models

from collections import namedtuple

#============== Model ============#

class Vgg19(torch.nn.Module):

	def __init__(self):
		super().__init__()
		# Gives list of layers with weights
		vgg_pretrained_features = models.vgg19(pretrained=True).features
		# Only these layers are used for style transfer
		self.layers_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
		self.used_layers_cnt = len(self.layers_names)
		self.content_feature_maps_index = 4;
		self.style_feature_maps_indices = [0, 1, 2, 3, 5]

		self.slice_range = [0,1,6,11,20,22,29]
		self.slices = [ torch.nn.Sequential() for i in range(self.used_layers_cnt) ]

		for slice_ind in range(self.used_layers_cnt):
			for layer_ind in range(self.slice_range[slice_ind], self.slice_range[slice_ind+1]):
				self.slices[slice_ind].add_module(str(layer_ind), vgg_pretrained_features[layer_ind])
		
		# do not use autograd for parameters
		for param in self.parameters():
			for param in self.parameters():
				param.requires_grad = False


	def forward(self, x):
		results = []
		for slice_ind in range(self.used_layers_cnt):
			x = self.slices[slice_ind](x)
			results.append(x)

		out = dict(zip(self.layers_names, results))
		return out
