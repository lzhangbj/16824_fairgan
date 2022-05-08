from enum import Enum
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from .helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from .restyle_e4e_encoders import EqualLinear


class GradualStyleBlock(Module):
	def __init__(self, in_c, out_c, spatial):
		super(GradualStyleBlock, self).__init__()
		self.out_c = out_c
		self.spatial = spatial
		num_pools = int(np.log2(spatial))
		modules = []
		modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
					nn.LeakyReLU()]
		for i in range(num_pools - 1):
			modules += [
				Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
				nn.LeakyReLU()
			]
		self.convs = nn.Sequential(*modules)
		self.linear = EqualLinear(out_c, out_c, lr_mul=1)

	def forward(self, x):
		x = self.convs(x)
		x = x.view(-1, self.out_c)
		x = self.linear(x)
		return x


class Encoder4Editing(Module):
	def __init__(self, num_layers=50, mode='ir_se'):
		super(Encoder4Editing, self).__init__()
		assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
		assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
		blocks = get_blocks(num_layers)
		if mode == 'ir':
			unit_module = bottleneck_IR
		elif mode == 'ir_se':
			unit_module = bottleneck_IR_SE
		self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
									  BatchNorm2d(64),
									  PReLU(64))
		modules = []
		for block in blocks:
			for bottleneck in block:
				modules.append(unit_module(bottleneck.in_channel,
										   bottleneck.depth,
										   bottleneck.stride))
		self.body = Sequential(*modules)

		self.styles = nn.ModuleList()
		self.style_count = 1

		for i in range(self.style_count):
			style = GradualStyleBlock(512, 512, 16)
			self.styles.append(style)

		self.load_pretrained()
		for param in self.parameters():
			param.requires_grad = False

	def load_pretrained(self):
		ckpt_path = './pretrained/e4e_fairface125_eric.pt'
		load_state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
		encoder_state_dict = dict()
		for key, val in load_state_dict.items():
			if 'encoder' in key:
				if 'styles.' in key:
					if 'styles.0' in key:
						encoder_state_dict[key.replace('encoder.', '')] = val
				elif 'latlayer' in key:
					pass
				else:
					encoder_state_dict[key.replace('encoder.', '')] = val
		self.load_state_dict(encoder_state_dict)

	def forward(self, x):
		x = self.input_layer(x)

		modulelist = list(self.body._modules.values())
		for i, l in enumerate(modulelist):
			x = l(x)
			if i == 6:
				c1 = x
			elif i == 20:
				c2 = x
			elif i == 23:
				c3 = x

		# Infer main W and duplicate it
		w0 = self.styles[0](c3)
		return w0