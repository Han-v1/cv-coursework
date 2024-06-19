#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import argparse
import torch
from datetime import datetime

from models import model_dict
from datasets.dataset import get_dataloaders
from trainer.utils import cal_params_ptflops, message
from configs.cfg import cfg
from trainer import Trainer

torch.backends.cudnn.benchmark = True  # 自动选择最佳的算法和配置，提高计算性能
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 配置设备


def main(cfg, resume, repath, ckpt_name):
	# -----实验名称----- #
	experiment_name = "{}_{}".format(cfg.base.model_name, cfg.dataset.type)
	if resume:
		save_path = os.path.join(experiment_name, repath)
	else:
		now_time = datetime.strftime(datetime.now(), '%y-%m-%d_%H-%M')  # 获取当前时间
		save_path = os.path.join(experiment_name, "r{}_e{}".format(now_time, cfg.base.epochs))
	print(message("experiment_name: {}".format(experiment_name), "INFO"))
	print(message("save_path: {}".format(save_path), "INFO"))

	# -----输出配置信息----- #
	print(message("CONFIG:\n{}".format(cfg.dump()), "INFO"))

	# -----获取数据----- #
	print(message("Loading datasets.", "START"))
	train_loader, val_loader = get_dataloaders(
			batch_size=cfg.base.batch_size,
			num_workers=cfg.dataset.num_workers,
			train_txt_path=cfg.dataset.train_txt_path,
			val_test_txt_path=cfg.dataset.val_txt_path,
			is_test=False
			)
	print(message("Dataset loaded successfully!", "END"))

	# -----加载模型----- #
	print(message("Loading original model", "START"))
	model = model_dict[cfg.base.model_name](num_classes=cfg.dataset.num_classes)
	print(message("Original model loaded successfully!", "END"))

	# -----并行运算----- #
	model = torch.nn.DataParallel(model.to(device))

	# -----计算浮点数 & 参数量----- #
	flops, params = cal_params_ptflops(model, (3, 256, 256))
	print(message('Computational complexity (FLOPs):  {:<8}'.format(flops), "INFO"))
	print(message('Number of parameters (Params):  {:<8}'.format(params), "INFO"))

	# -----初始化训练器----- #
	trainer = Trainer(save_path, model, train_loader, val_loader, ckpt_name, cfg)
	trainer.start_train_val(resume=resume, flops=flops, params=params)


if __name__ == "__main__":

	# -----接收参数----- #
	parser = argparse.ArgumentParser("training for model.")
	parser.add_argument("--config", type=str, default="./configs/SFIConvResNet26.yaml")
	parser.add_argument("--resume", type=bool, default=False)
	parser.add_argument("--repath", type=str, default="r24-06-06_16-41_e100")
	parser.add_argument("--ckpt_name", type=str, default="epoch_9.pth")
	args = parser.parse_args()

	# -----整合参数----- #
	cfg.merge_from_file(args.config)  # 合并配置文件
	cfg.freeze()  # 锁定配置，变为只读状态

	# -----主程序----- #
	main(cfg, args.resume, args.repath, args.ckpt_name)
