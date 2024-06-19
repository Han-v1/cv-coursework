#!/usr/bin/python
# -*- coding: UTF-8 -*-

from yacs.config import CfgNode as CN

# from .utils import msg


# def show_cfg(configs):
# 	dump_cfg = CN()
# 	dump_cfg.base = configs.base
# 	dump_cfg.datasets = configs.datasets
# 	dump_cfg.distiller = configs.distiller
# 	dump_cfg.optimizer = configs.optimizer
# 	dump_cfg.scheduler = configs.scheduler
# 	dump_cfg.ReviewKD = configs.ReviewKD
# 	print(msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))


cfg = CN()

# Base
cfg.base = CN()
cfg.base.notes = ""
cfg.base.model_name = "resnet18"
cfg.base.batch_size = 64
cfg.base.epochs = 100
cfg.base.log_path = "./output"
cfg.base.save_frequency = 4

# Dataset
cfg.dataset = CN()
cfg.dataset.type = "ff_c23"
cfg.dataset.train_txt_path = "./data/ff_c23/train.txt"  # 训练集目录文本 地址
cfg.dataset.val_txt_path = "./data/ff_c23/val.txt"  # 验证集目录文本 地址
cfg.dataset.num_classes = 2
cfg.dataset.num_workers = 2

# Optimizer
cfg.optimizer = CN()
cfg.optimizer.type = "Adam"
cfg.optimizer.lr = 0.0001
cfg.optimizer.momentum = 0.9
cfg.optimizer.weight_decay = 0.0001

# Scheduler
cfg.scheduler = CN()
cfg.scheduler.type = "ExponentialLR"
cfg.scheduler.step_size = 1
cfg.scheduler.gamma = 0.5
cfg.scheduler.T_max = 150
cfg.scheduler.last_epoch = -1
