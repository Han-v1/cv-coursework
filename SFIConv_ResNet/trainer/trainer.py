import os
import time

from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset import mixup_criterion, mixup_data
from .utils import (message, AverageMeter, save_checkpoint, load_checkpoint, train_record, val_record)


class Trainer:
	def __init__(self, save_path, model, train_loader, val_loader, ckpt_name, cfg):
		# -----传入参数----- #
		self.save_path = save_path
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.ckpt_name = ckpt_name
		self.cfg = cfg

		self.train_count = 0
		self.val_count = 0

		self.optimizer = self.init_optimizer(cfg)  # 优化器
		self.scheduler = self.init_scheduler(cfg)  # 学习率调度器
		self.loss_function = CrossEntropyLoss()  # 交叉熵损失（多分类）
		self.best_acc = -1

		# -----初始化路径----- #
		self.log_path = str(os.path.join(cfg.base.log_path, save_path))
		self.record_path = os.path.join(self.log_path, "record")
		self.ckpt_path = os.path.join(self.log_path, "ckpt")
		if not os.path.exists(self.record_path):
			os.makedirs(self.record_path)
		if not os.path.exists(self.ckpt_path):
			os.makedirs(self.ckpt_path)

		# -----TensorBoard----- #
		self.tf_writer = SummaryWriter(str(os.path.join(self.log_path, "events")))

	def start_train_val(self, resume=False, flops=None, params=None):
		epoch = 1
		configs_msg = "\n{}".format(self.cfg.dump())
		flops_msg = '\nComputational complexity (FLOPs):  {:<8}'.format(flops)
		params_msg = '\nNumber of parameters (Params):  {:<8}'.format(params)
		with open(os.path.join(self.record_path, "finally.txt"), "a") as w:
			w.write("\n" + "=" * 20 + "CONFIG" + "=" * 20)
			w.write(configs_msg)
			w.write("=" * 20 + "FLOPs & Params" + "=" * 20)
			w.write(flops_msg)
			w.write(params_msg)

		# -----恢复断点----- #
		if resume:
			state = load_checkpoint(os.path.join(self.ckpt_path, self.ckpt_name))
			epoch = state["epoch"] + 1
			self.model.load_state_dict(state["model"])
			self.optimizer.load_state_dict(state["optimizer"])
			self.scheduler.last_epoch = state["epoch"]
			self.best_acc = state["acc"]
		# for param_group in self.optimizer.param_groups:
		# 	print('lr', param_group['lr'])
		# 	param_group['lr'] *= 0.5
		# 	print('new_lr', param_group['lr'])
		# self.scheduler.load_state_dict(state["scheduler"])

		# -----Epoch----- #
		start_time = time.time()
		while epoch <= self.cfg.base.epochs:
			self.train_val_epoch(epoch)
			epoch += 1
		total_time = int(time.time() - start_time)
		hours, remainder = divmod(total_time, 3600)
		minutes, seconds = divmod(remainder, 60)
		time_hms = str(hours) + ':' + str(minutes) + ':' + str(seconds)
		time_msg = "all_time: {}".format(time_hms)
		print(message(time_msg, "TIME"))

		# -----输出最终结果----- #
		best_acc_msg = "best_acc: {:.5f}".format(float(self.best_acc))
		print(message(best_acc_msg, "ACC"))
		with open(os.path.join(self.record_path, "finally.txt"), "a") as w:
			w.write("\n" + "=" * 20 + "Total Time" + "=" * 20 + "\n")
			w.write(time_msg)
			w.write("\n" + "=" * 20 + "Best Acc" + "=" * 20 + "\n")
			w.write(best_acc_msg)
			w.write("\n" + "=" * 50)

	def train_val_epoch(self, epoch):
		# -----相关指标----- #
		train_metrics = {
			"time_train": AverageMeter(),
			"loss_train": AverageMeter(),
			"acc_train": AverageMeter(),
			"auc_train": float(),
			"auc_train_target": list(),
			"auc_train_prediction": list(),
			}
		val_metrics = {
			"time_val": AverageMeter(),
			"loss_val": AverageMeter(),
			"acc_val": AverageMeter(),
			"auc_val": float(),
			"auc_val_target": list(),
			"auc_val_prediction": list(),
			}

		'''# -----训练----- #'''
		self.model.train()
		num_train_loader = len(self.train_loader)
		pbar_train = tqdm(range(num_train_loader))  # train进度条
		for idx_train, data_train in enumerate(self.train_loader):
			msg_train = self.train_iter(data_train, epoch, train_metrics)
			self.train_count += 1
			pbar_train.set_description(message(msg_train, "TRAIN"))
			pbar_train.update()
		pbar_train.close()
		self.scheduler.step()  # 更新学习率

		train_metrics["auc_train"] = roc_auc_score(train_metrics["auc_train_target"], train_metrics["auc_train_prediction"])
		print(message("auc_train: {:.5f}".format(train_metrics["auc_train"]), mode="TRAIN"))
		# -----输出训练记录----- #
		self.tf_writer.add_scalars(main_tag="train/train_auc", tag_scalar_dict={"train_auc": train_metrics["auc_train"]}, global_step=epoch)
		self.tf_writer.flush()
		train_dict = OrderedDict(
				{
					"train_acc": train_metrics["acc_train"].avg,
					"train_auc": train_metrics["auc_train"],
					"train_loss": train_metrics["loss_train"].avg,
					}
				)
		train_record(self.scheduler.get_last_lr()[0], epoch, self.cfg.base.epochs, train_dict, self.record_path)

		'''# -----验证----- #'''
		self.model.eval()
		num_val_loader = len(self.val_loader)
		pbar_val = tqdm(range(num_val_loader))  # val进度条
		with torch.no_grad():  # PyTorch会暂时停止自动求导
			for idx_val, data_val in enumerate(self.val_loader):
				msg_val = self.val_iter(data_val, epoch, val_metrics)
				self.val_count += 1
				pbar_val.set_description(message(msg_val, "VAL"))
				pbar_val.update()
		pbar_val.close()

		val_metrics["auc_val"] = roc_auc_score(val_metrics["auc_val_target"], val_metrics["auc_val_prediction"])
		print(message("auc_val: {:.5f}".format(val_metrics["auc_val"]), mode="VAL"))
		# -----输出验证记录----- #
		self.tf_writer.add_scalars(main_tag="val/val_auc", tag_scalar_dict={"val_auc": val_metrics["auc_val"]}, global_step=epoch)
		self.tf_writer.flush()
		val_dict = OrderedDict(
				{
					"val_acc": val_metrics["acc_val"].avg,
					"val_auc": val_metrics["auc_val"],
					"val_loss": val_metrics["loss_val"].avg,
					}
				)
		val_record(self.scheduler.get_last_lr()[0], epoch, self.cfg.base.epochs, val_dict, self.record_path)

		# -----输出训练&验证对比记录----- #
		self.tf_writer.add_scalars(main_tag="compare/loss", tag_scalar_dict={"train_loss": train_dict["train_loss"], "val_loss": val_dict["val_loss"]}, global_step=epoch)
		self.tf_writer.add_scalars(main_tag="compare/acc", tag_scalar_dict={"train_acc": train_dict["train_acc"], "val_acc": val_dict["val_acc"]}, global_step=epoch)
		self.tf_writer.add_scalars(main_tag="compare/auc", tag_scalar_dict={"train_auc": train_dict["train_auc"], "val_auc": val_dict["val_auc"]}, global_step=epoch)
		self.tf_writer.flush()

		# -----保存模型----- #
		state = {
			"epoch": epoch,
			"model": self.model.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"scheduler": self.scheduler.state_dict(),
			"acc": val_metrics["acc_val"].avg,
			}
		save_checkpoint(state, os.path.join(self.ckpt_path, "latest.pth"))
		if epoch % self.cfg.base.save_frequency == 0:
			save_checkpoint(state, os.path.join(self.ckpt_path, "epoch_{}.pth".format(epoch)))

		# -----更新最优----- #
		if val_metrics["acc_val"].avg >= self.best_acc:
			self.best_acc = val_metrics["acc_val"].avg
			save_checkpoint(state, os.path.join(self.ckpt_path, "best.pth"))

	def train_iter(self, data_train, epoch, train_metrics):
		self.optimizer.zero_grad()

		train_start_time = time.time()

		img_train, tar_train = data_train
		img_train = img_train.float().cuda(non_blocking=True)
		tar_train = tar_train.cuda(non_blocking=True)

		# pre_train = self.model(img_train)  # forward

		data, y_a, y_b, lam = mixup_data(img_train, tar_train, 0.5)
		pre_train = self.model(img_train)
		train_loss = mixup_criterion(self.loss_function, pre_train, y_a, y_b, lam)

		train_time = time.time() - train_start_time

		# -----计算train损失----- #
		# train_loss = self.loss_function(pre_train, tar_train)

		train_loss.backward()  # backward
		self.optimizer.step()

		# -----预测结果 正则化----- #
		pre_train = F.softmax(pre_train, dim=1)

		# -----预测结果 非0即1 并计算 acc----- #
		_, pre_acc = torch.max(pre_train.data, dim=1)  # max_values, max_indices = torch.max(x, dim=1)

		# -----acc----- #
		pre_acc = pre_acc.detach().cpu().numpy()
		tar_train = tar_train.detach().cpu().numpy()
		train_acc = accuracy_score(tar_train, pre_acc)

		pre_train = pre_train.detach().cpu().numpy()

		# -----记录数据----- #
		bz = img_train.size(0)  # batch_size
		train_metrics["time_train"].update(train_time)
		train_metrics["loss_train"].update(train_loss)
		train_metrics["acc_train"].update(train_acc)
		train_metrics["auc_train_target"].extend(tar_train.tolist())
		train_metrics["auc_train_prediction"].extend(pre_train[:, 1].tolist())

		# -----输出信息----- #
		self.tf_writer.add_scalars(main_tag="train/train_loss", tag_scalar_dict={"train_loss": train_metrics["loss_train"].avg}, global_step=self.train_count)
		self.tf_writer.add_scalars(main_tag="train/train_acc", tag_scalar_dict={"train_acc": train_metrics["acc_train"].avg}, global_step=self.train_count)
		self.tf_writer.flush()

		train_msg = "[Epoch:{}|{}] [time:{:.3f} | loss:{:.5f} | acc:{:.5f}] ".format(
				epoch,
				self.cfg.base.epochs,
				train_metrics["time_train"].avg,
				train_metrics["loss_train"].avg,
				train_metrics["acc_train"].avg,
				)
		return train_msg

	def val_iter(self, data_val, epoch, val_metrics):
		val_start_time = time.time()

		img_val, tar_val = data_val
		img_val = img_val.float().cuda(non_blocking=True)
		tar_val = tar_val.cuda(non_blocking=True)

		pre_val = self.model(img_val)  # forward

		val_time = time.time() - val_start_time

		# -----计算val损失----- #
		val_loss = self.loss_function(pre_val, tar_val)

		# -----预测结果 正则化----- #
		pre_val = F.softmax(pre_val, dim=1)

		# -----预测结果 非0即1 并计算 acc----- #
		_, pre_acc = torch.max(pre_val.data, dim=1)  # max_values, max_indices = torch.max(x, dim=1)

		# -----acc----- #
		pre_acc = pre_acc.detach().cpu().numpy()
		tar_val = tar_val.detach().cpu().numpy()
		val_acc = accuracy_score(tar_val, pre_acc)

		pre_val = pre_val.detach().cpu().numpy()

		# -----模型的预测----- #
		val_metrics["time_val"].update(val_time)
		val_metrics["loss_val"].update(val_loss)
		val_metrics["acc_val"].update(val_acc)
		val_metrics["auc_val_target"].extend(tar_val.tolist())
		val_metrics["auc_val_prediction"].extend(pre_val[:, 1].tolist())

		# -----输出信息----- #
		self.tf_writer.add_scalars(main_tag="val/val_loss", tag_scalar_dict={"val_loss": val_metrics["loss_val"].avg}, global_step=self.val_count)
		self.tf_writer.add_scalars(main_tag="val/val_acc", tag_scalar_dict={"val_acc": val_metrics["acc_val"].avg}, global_step=self.val_count)
		self.tf_writer.flush()

		val_msg = "[Epoch:{}|{}] [time:{:.3f} | loss:{:.5f} | acc:{:.5f}] ".format(
				epoch,
				self.cfg.base.epochs,
				val_metrics["time_val"].avg,
				val_metrics["loss_val"].avg,
				val_metrics["acc_val"].avg,
				)
		return val_msg

	def init_optimizer(self, cfg):
		if cfg.optimizer.type == "SGD":
			optimizer = optim.SGD(
					self.model.parameters(),
					lr=cfg.optimizer.lr,
					momentum=cfg.optimizer.momentum,
					weight_decay=cfg.optimizer.weight_decay,
					)
		elif cfg.optimizer.type == "Adam":
			optimizer = optim.Adam(
					self.model.parameters(),
					lr=cfg.optimizer.lr,
					weight_decay=cfg.optimizer.weight_decay,
					)
		else:
			raise NotImplementedError(cfg.optimizer.type)
		return optimizer

	def init_scheduler(self, cfg):
		if cfg.scheduler.type == "StepLR":
			scheduler = StepLR(
					optimizer=self.optimizer,
					step_size=cfg.scheduler.step_size,
					gamma=cfg.scheduler.gamma,
					last_epoch=cfg.scheduler.last_epoch
					)
		elif cfg.scheduler.type == "ExponentialLR":
			scheduler = ExponentialLR(
					optimizer=self.optimizer,
					gamma=cfg.scheduler.gamma,
					last_epoch=cfg.scheduler.last_epoch
					)
		elif cfg.scheduler.type == "CosineAnnealingLR":
			scheduler = CosineAnnealingLR(
					optimizer=self.optimizer,
					T_max=self.cfg.scheduler.T_max,
					eta_min=1e-7,
					last_epoch=cfg.scheduler.last_epoch
					)
		else:
			raise NotImplementedError(cfg.scheduler.type)
		return scheduler
