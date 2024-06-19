import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class BaseDataset(Dataset):
	def __init__(self, txt_path, train_transform=None, valid_transform=None):
		lines = open(txt_path, 'r')
		image_label = []
		for line in lines:
			line = line.rstrip()  # 删除字符串末尾所有连续的空白字符（包括空格、制表符和换行符）。
			words = line.split()  # 返回一个字符串列表[img_path, label]
			image_label.append((words[0], int(words[1])))  # img_path, label

		self.data = image_label  # 生成全局列表
		self.train_transform = train_transform
		self.valid_transform = valid_transform

	def __getitem__(self, index):
		img_path, label = self.data[index]
		img = Image.open(img_path).convert('RGB')
		# transform
		if self.train_transform is not None:
			img = self.train_transform(img)
		if self.valid_transform is not None:
			img = self.valid_transform(img)

		return img, label

	def __len__(self):
		return len(self.data)


def get_ff_train_transform():
	return transforms.Compose(
			[
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(10),
				transforms.RandomPerspective(),
				transforms.ToTensor(),
				transforms.Normalize(
						mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225]
						)
				]
			)


def get_ff_val_test_transform():
	return transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize(
						mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225]
						)
				]
			)


def get_dataloaders(batch_size, num_workers, train_txt_path=None, val_test_txt_path=None, is_test=False):
	if is_test:
		val_test_transform = get_ff_val_test_transform()
		val_test_data = BaseDataset(txt_path=val_test_txt_path, valid_transform=val_test_transform)
		val_test_loader = DataLoader(val_test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)

		return val_test_loader
	else:
		train_transform = get_ff_train_transform()
		train_data = BaseDataset(txt_path=train_txt_path, train_transform=train_transform)
		train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

		val_test_transform = get_ff_val_test_transform()
		val_test_data = BaseDataset(txt_path=val_test_txt_path, valid_transform=val_test_transform)
		val_test_loader = DataLoader(val_test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)

		return train_loader, val_test_loader


def mixup_data(x, y, alpha=0.5, use_cuda=False):
	"""Returns mixed inputs, pairs of targets, and lambda"""
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	a = criterion(pred, y_a)
	b = criterion(pred, y_b)
	losses = []
	try:
		for i in range(len(a)):
			losses.append(lam * a[i] + (1 - lam) * b[i])
	except:
		return lam * a + (1 - lam) * b
	return losses


if __name__ == '__main__':
	train, valid, num = get_dataloaders(10, 2, False)
	print(train.dataset)
	print(num)
