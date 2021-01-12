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
		self.content_feature_maps_index = 4;
		self.style_feature_maps_indices = [0, 1, 2, 3, 5]

		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()
		self.slice6 = torch.nn.Sequential()

		for x in range(1):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(1, 6):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(6, 11):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(11, 20):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(20, 22):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		for x in range(22, 29):
			self.slice6.add_module(str(x), vgg_pretrained_features[x])
		
		# do not use autograd for parameters
		for param in self.parameters():
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, x):
		results = []
		x = self.slice1(x)
		results.append(x)
		x = self.slice2(x)
		results.append(x)
		x = self.slice3(x)
		results.append(x)
		x = self.slice4(x)
		results.append(x)
		x = self.slice5(x)
		results.append(x)
		x = self.slice6(x)
		results.append(x)

		out = dict(zip(self.layers_names, results))
		return out
