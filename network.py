import torch
import torchvision.models as models

from collections import namedtuple

#============== Model ============#

class Vgg19(torch.nn.Module):

	def __init__(self):
		super().__init__()
		# Only these layers are used for style transfer
		self.layers_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
		self.used_layers_cnt = len(self.layers_names)
		self.content_feature_maps_index = 4;
		self.style_feature_maps_indices = [0, 1, 2, 3, 5]

		self.slice_range = [0,1,6,11,20,22,29]
		self.all_slices = [ torch.nn.Sequential() for i in range(self.used_layers_cnt) ]
		
		self.construct_slices()

		# do not use autograd for parameters
		for param in self.parameters():
			for param in self.parameters():
				param.requires_grad = False


	def forward(self, x):
		results = []
		for slice_ind in range(self.used_layers_cnt):
			x = self.all_slices[slice_ind](x)
			results.append(x)

		out = dict(zip(self.layers_names, results))
		return out


	def construct_slices(self):
		vgg_pretrained_features = models.vgg19(pretrained=True).features
		
		for slice_ind in range(self.used_layers_cnt):
			for layer_ind in range(self.slice_range[slice_ind], self.slice_range[slice_ind+1]):
				if vgg_pretrained_features[layer_ind].__class__.__name__ == 'MaxPool2d':
					self.all_slices[slice_ind].add_module(str(layer_ind), torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
				else:
					self.all_slices[slice_ind].add_module(str(layer_ind), vgg_pretrained_features[layer_ind])
