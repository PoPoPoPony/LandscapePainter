import torch.nn as nn
import torch

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
	norm = torch.norm(input, 2, axis, True)
	output = torch.div(input, norm)
	return output


class SEModule(nn.Module):
	def __init__(self, channels, reduction) -> None:
		super(SEModule, self).__init__()
		self.model = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, channels // reduction, 1, 1, 0, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels // reduction, channels, 1, 1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		model_input = x
		x = self.model(x)

		return model_input * x


class bottleneck_IR_SE(nn.Module):
	def __init__(self, in_channel, out_channel, stride) -> None:
		super(bottleneck_IR_SE, self).__init__()
		if in_channel == out_channel:
			self.shortcut_layer = nn.MaxPool2d(1, stride)
		else:
			self.shortcut_layer = nn.Sequential(
				nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
				nn.BatchNorm2d(out_channel)
			)
		self.model = nn.Sequential(
			nn.BatchNorm2d(in_channel),
			nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
			nn.PReLU(out_channel),
			nn.Conv2d(out_channel, out_channel, 3, stride, 1, bias=False),
			nn.BatchNorm2d(out_channel),
			SEModule(out_channel, 16)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		x = self.model(x)

		return x + shortcut



def get_block(in_channel, out_channel, nums):
	blocks = [bottleneck_IR_SE(in_channel, out_channel, 2)]
	for _ in range(nums-1):
		blocks.append(bottleneck_IR_SE(out_channel, out_channel, 1))

	return blocks


def get_blocks():
	blocks = [
		*get_block(64, 64, 3),
		*get_block(64, 128, 4),
		*get_block(128, 256, 14),
		*get_block(256, 512, 3),
	]

	return blocks


class bottleneck_IR(nn.Module):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = nn.MaxPool2d(1, stride)
		else:
			self.shortcut_layer = nn.Sequential(
				nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				nn.BatchNorm2d(depth)
			)
		self.res_layer = nn.Sequential(
			nn.BatchNorm2d(in_channel),
			nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), nn.PReLU(depth),
			nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), nn.BatchNorm2d(depth)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut