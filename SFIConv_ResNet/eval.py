import os
import argparse

import numpy as np
import torch
import argparse
from datetime import datetime
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm

from datasets.dataset import get_dataloaders
from models import model_dict
from trainer.utils import AverageMeter, cal_params_ptflops, load_checkpoint, message

torch.backends.cudnn.benchmark = True  # 自动选择最佳的算法和配置，提高计算性能
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 配置设备


def test_iter(test_model, data_val, metrics_test):
	test_start_time = time.time()

	img_test, tar_test = data_val
	img_test = img_test.float().cuda(non_blocking=True)
	tar_test = tar_test.cuda(non_blocking=True)

	pre_test, _ = test_model(img_test)  # forward

	test_time = time.time() - test_start_time

	# -----预测结果 正则化----- #
	pre_test = F.softmax(pre_test, dim=1)

	# -----预测结果 非0即1 并计算 acc----- #
	_, pre_acc = torch.max(pre_test.data, dim=1)  # max_values, max_indices = torch.max(x, dim=1)

	# -----acc----- #
	pre_acc = pre_acc.detach().cpu().numpy()
	tar_test = tar_test.detach().cpu().numpy()
	test_acc = accuracy_score(tar_test, pre_acc)

	pre_test = pre_test.detach().cpu().numpy()

	# -----模型的预测----- #
	metrics_test["time_test"].update(test_time)
	metrics_test["acc_test"].update(test_acc)
	metrics_test["auc_test_target"].extend(tar_test.tolist())
	metrics_test["auc_test_prediction"].extend(pre_test[:, 1].tolist())

	test_msg = "[time:{:.3f} | acc:{:.5f}] ".format(metrics_test["time_test"].avg, metrics_test["acc_test"].avg)
	return test_msg


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", type=str, default="SpaIFNet26_v2")
	parser.add_argument("--num_classes", type=int, default=2)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--test_txt_path", type=str, default="./datasets/ff_c23_official_split/test.txt")
	parser.add_argument("--pretrain_model_path", type=str, default="./pretrain_ckpts/SpaIFNet26_v2/acc_941147_ff_c23.pth")
	args = parser.parse_args()

	# -----获取数据----- #
	print(message("Loading datasets.", "START"))
	test_loader = get_dataloaders(
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			val_test_txt_path=args.test_txt_path,
			is_test=True
			)
	print(message("Dataset loaded successfully!", "END"))

	# -----加载模型----- #
	print(message("Loading original model", "START"))
	model = model_dict[args.model_name][0](num_classes=args.num_classes)
	model.to(device)
	model = torch.nn.DataParallel(model.cuda())  # 多GPU并行运算
	state = load_checkpoint(args.pretrain_model_path)
	print("best acc:{}".format(state["acc"]))
	model.load_state_dict(state["model"])
	print(message("Original model loaded successfully!", "END"))

	model.to(device)
	model = torch.nn.DataParallel(model.cuda())  # 多GPU并行运算

	# -----计算浮点数 & 参数量----- #
	flops, params = cal_params_ptflops(model, (3, 256, 256))
	print(message('Computational complexity (FLOPs):  {:<8}'.format(flops), "INFO"))
	print(message('Number of parameters (Params):  {:<8}'.format(params), "INFO"))

	# -----测试----- #
	test_metrics = {
		"time_test": AverageMeter(),
		"acc_test": AverageMeter(),
		"auc_test": float(),
		"auc_test_target": list(),
		"auc_test_prediction": list(),
		"eer_test": float(),
		}
	model.eval()
	test_length = len(test_loader)
	pbar_test = tqdm(range(test_length))
	with torch.no_grad():
		for data_test in test_loader:
			msg_test = test_iter(model, data_test, test_metrics)
			pbar_test.set_description(message(msg_test, "TEST"))
			pbar_test.update()
	pbar_test.close()

	test_metrics["auc_test"] = roc_auc_score(test_metrics["auc_test_target"], test_metrics["auc_test_prediction"])

	fpr, tpr, thresholds = roc_curve(test_metrics["auc_test_target"], test_metrics["auc_test_prediction"])
	test_metrics["eer_test"] = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]  # 计算EER

	print(message("acc_test: {:.5f}".format(test_metrics["acc_test"].avg), mode="TEST"))
	print(message("auc_test: {:.5f}".format(test_metrics["auc_test"]), mode="TEST"))
	print(message("eer_test: {:.5f}".format(test_metrics["eer_test"]), mode="TEST"))
