import os
import torch
import numpy as np
from thop import profile, clever_format
from ptflops import get_model_complexity_info


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self, val=0., avg=0., sum=0, count=0):
		self.val = val
		self.avg = avg
		self.sum = sum
		self.count = count

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def message(msg, mode="INFO"):
	"""
	黑色 30（高亮）90
    红色 31（高亮）91
    绿色 32（高亮）92
    黄色 33（高亮）93
    蓝色 34（高亮）94
    紫色 35（高亮）95
    青色 36（高亮）96
    白色 37（高亮）97
    """
	color_map = {
		"START": 97,
		"END": 97,
		"INFO": 96,
		"TRAIN": 94,
		"VAL": 92,
		"TEST": 95,
		"DEBUG": 93,
		"ACC": 95,
		"TIME": 95,
		}
	msg = "\033[{}m[{:<5}] {}\033[0m".format(color_map[mode], mode, msg)
	return msg


def load_checkpoint(path):
	with open(path, "rb") as f:
		return torch.load(f, map_location="cpu")


def save_checkpoint(obj, path):
	with open(path, "wb") as f:
		torch.save(obj, f)


def train_record(lr, epoch, total_epoch, train_log, record_path):
	# -----train.txt-----#
	with open(os.path.join(record_path, "train.txt"), "a") as w_train:
		line = "[Train] [Epoch: {:>3}|{:<3}] [lr: {:.10f}".format(epoch, total_epoch, float(lr))
		for k, v in train_log.items():
			line += ("] [{}: {:.5f}".format(k, v))
		line += ']\n'
		w_train.writelines(line)


def val_record(lr, epoch, total_epoch, val_log, record_path):
	# -----val.txt-----#
	with open(os.path.join(record_path, "val.txt"), "a") as w_val:
		line = "[Val] [Epoch: {:>3}|{:<3}] [lr: {:.10f}".format(epoch, total_epoch, float(lr))
		for k, v in val_log.items():
			line += ("] [{}: {:.5f}".format(k, v))
		line += ']\n'
		w_val.writelines(line)


def params_count(model):
	"""
	Compute the parameters.
	"""
	return np.sum([p.numel() for p in model.parameters()]).item()


def cal_params_thop(model, tensor):
	"""
	Using thop to compute the parameters, FLOPs
	tensor: torch.randn(1, 3, 256, 256)
	"""
	flops, params = profile(model, inputs=(tensor,))
	flops, params = clever_format([flops, params], '%.3f')
	return flops, params


def cal_params_ptflops(model, shape):
	"""
	Using ptflops to compute the parameters, FLOPs
	shape: (3, 256, 256)

	model: 要分析的模型。
	shape: 输入数据的尺寸，用于模拟前向传播过程以估算FLOPs。
	as_strings: 如果为True，则返回的FLOPs和参数量将是易于阅读的字符串形式（如带有单位‘M’表示百万），否则为原始数值。
	print_per_layer_stat: 如果为True，会打印出每层网络的参数量和FLOPs统计信息。
	verbose: 设置为True时，会在控制台输出详细的计算过程信息。
	"""
	with torch.cuda.device(0):
		flops, params = get_model_complexity_info(model, shape, as_strings=True, print_per_layer_stat=False, verbose=False)
	return flops, params
