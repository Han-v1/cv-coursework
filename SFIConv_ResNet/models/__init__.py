from models.resnet import (resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2)
from models.SFIConvResNet import (SFIConvResNet26, SFIConvResNet50, SFIConvResNet101, SFIConvResNet152)

model_dict = {
	# ResNet
	"resnet18": resnet18,
	"resnet34": resnet34,
	"resnet50": resnet50,
	"resnet101": resnet101,
	"resnet152": resnet152,
	"resnext50_32x4d": resnext50_32x4d,
	"resnext101_32x8d": resnext101_32x8d,
	"wide_resnet50_2": wide_resnet50_2,
	"wide_resnet101_2": wide_resnet101_2,

	# SFIConvResNet
	"SFIConvResNet26": SFIConvResNet26,
	"SFIConvResNet50": SFIConvResNet50,
	"SFIConvResNet101": SFIConvResNet101,
	"SFIConvResNet152": SFIConvResNet152,
	}
