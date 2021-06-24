import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import time
import torch
import random
import numpy as np

from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def train_attack_model(TARGET_PATH, ATTACK_PATH, classes, device, target_model, train_loader, test_loader, epoch, loss, optimizer, dataset_type, mode, get_attack_set, r):
	input_classes = get_gradient_size(target_model)
	attack_model = OverlearningAttackModel(input_classes=input_classes, output_classes=classes)
	ATTACK_SETS = TARGET_PATH + "ol_epoch_" + str(epoch) + "_" + loss + "_" + optimizer + "_mode" + str(args.mode)
	TARGET_PATH = TARGET_PATH + "target_epoch_" + str(epoch) + "_" + loss + "_" + optimizer + ".pth"
	MODELS_PATH = ATTACK_PATH + "attack_epoch_" + str(epoch) + "_" + loss + "_" + optimizer + ".pth"
	RESULT_PATH = ATTACK_PATH + "attack_epoch_" + str(epoch) + "_" + loss + "_" + optimizer + ".p"

	attack = attack_training(train_loader, test_loader, attack_model, target_model, device, TARGET_PATH, r)

	# if get_attack_set:
	# 	attack.delete_pickle(ATTACK_SETS)
	# 	attack.prepare_dataset(ATTACK_SETS)

	for epoch in range(300):
		print("<======================= Epoch " + str(epoch+1) + " =======================>")
		print("attack training")

		res_attack_train = attack.train(epoch, RESULT_PATH, dataset_type)
		print("attack testing")
		res_attack_test = attack.test(epoch, RESULT_PATH, dataset_type)

	attack.saveModel(MODELS_PATH)
	print("Saved Attack Model")
	print("Finished!!!")


	return res_attack_train, res_attack_test

def count_data(num_classes, dataset):
	data_list_1 = [0 for i in range(num_classes[0])]
	data_list_2 = [0 for i in range(num_classes[1])]

	for _, [num0, num1] in tqdm(dataset):
		data_list_1[num0] += 1
		data_list_2[num1] += 1

	print(data_list_1)
	print(data_list_2)

	data_list_2.sort(reverse=True)

	result = data_list_2[0]/(np.array(data_list_2).sum())*100.

	print('%.2f%%' % (result))

def get_model(model_name, num_classes):
	if model_name == "alexnet":
		from models.overlearning import AlexNet
		blackbox_oracle = AlexNet(num_classes=num_classes)
		
	elif model_name == "resnet18":
		from models.overlearning import resnet18
		blackbox_oracle = resnet18(num_classes=num_classes)

	elif model_name == "vgg19":
		from models.overlearning import vgg19_bn
		blackbox_oracle = vgg19_bn(num_classes=num_classes)

	elif model_name == "xception":
		from models.overlearning import xception
		blackbox_oracle = xception(num_classes=num_classes)

	elif model_name == "vgg11":
		from models.overlearning import vgg11_bn
		blackbox_oracle = vgg11_bn(num_classes=num_classes)

	elif model_name == "CNN":
		from models.overlearning import Simple_CNN
		blackbox_oracle = Simple_CNN(num_classes=num_classes)

	else:
		sys.exit("we have not supported this model yet! :((((")

	return blackbox_oracle

def get_gradient_size(model):
	gradient_list = reversed(list(model.named_parameters()))
	for name, parameter in gradient_list:
		if 'weight' in name:
			input_size = parameter.shape[1]
			break

	return input_size

def str_to_bool(string):
    if isinstance(string, bool):
       return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('-g', '--gpu', type=str_to_bool, default=True, 
						help='Choose if or not using GPU (default: yes)')

	parser.add_argument('-d', '--dataset', type=str, default='UTKFace', 
						help='Choose one Dataset, UTKFace or CelebA (default: UTKFace)')

	parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')

	parser.add_argument('-m', '--model', type=str, default="alexnet", 
						help='Choose one model (default: alexnet)')

	parser.add_argument('-a', '--attribute', type=str, default=None, 
						help='Choose one attribute for UTKFace(age, gender or race) and CelebA(landmarks, attr, identity or bbox, but we suggest using the attr to do membership inference attacks) (default: race)')

	parser.add_argument('-md', '--mode', type=int, default=0,
						help='you can choose the mode of membership inference attack. 0 means you can provide the shadow dataset for us, 1 means you can offer partial training dataset. 2 means whitebox attack(default: 0)')

	parser.add_argument('-dp', '--DP', type=str_to_bool, default=False,
						help='if use differential privacy model, defualt no')

	parser.add_argument('-c', '--checkpoint', type=str_to_bool, default=True,
						help='choose to save model in different epochs')

	parser.add_argument('-dl', '--distill', type=str_to_bool, default=False,
						help='where or not test a distillation model')

	parser.add_argument('-ne', '--noise', type=float, default=1.0,
						help='choose noise for dp model')

	parser.add_argument('-nm', '--norm', type=float, default=1.0,
						help='choose norm for dp model')

	parser.add_argument('-dpt', '--DP_type', type=int, default=1,
						help='choose noise and norm for dp model')

	parser.add_argument('-da', '--delta', type=float, default=1e-5,
						help='choose delta for dp model')

	parser.add_argument('-r', '--round', type=int, default=4,
						help='different rounds of attack model and get the average results')

	parser.add_argument('-gas', '--get_attack_set', type=str_to_bool, default=False,
						help='whether or not get attack set before train attack model, if false using the downloaded results')

	args = parser.parse_args()

	attributes = args.attribute.split('_')

	if args.gpu and torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")

	if args.checkpoint:
		checkpoint = [10, 20, 50, 100, 200, 300]
	else:
		checkpoint = [300]

	if args.DP:
		if args.DP_type == 0:
			noise = {"UTKFace": 1.3, "celeba": 0.9}
			norm = {"UTKFace": 1.5, "celeba": 1.5}

		elif args.DP_type == 1:
			noise = {"UTKFace": 1.5, "celeba": 0.8}
			norm = {"UTKFace": 2.0, "celeba": 2.0}

		else:
			sys.exit("we have not supported this DP mode! hahaha")

		loss = str(noise[args.dataset])
		optimizer = str(norm[args.dataset])

	else:
		loss = "CrossEntropyLoss"
		optimizer = "SGD"

	# get data set
	num_classes, target_train, target_test, shadow_train, shadow_test, _, _ = prepare_dataset(args.model, args.dataset, args.batch_size, attributes)

	# mkdir path
	TARGET_PATH = "../data/model/target/" + args.dataset + "/" + args.model + "/"
	ATTACK_PATH = "../data/model/attack/round" + str(args.round) + "/" + args.dataset + "/" + args.model + "/ol/mode" + str(args.mode) + "/"

	if args.distill:
		TARGET_PATH = TARGET_PATH + "distill/"
		ATTACK_PATH = ATTACK_PATH + "distill/"
		target_model = get_model("vgg11", num_classes[0])
	
	elif args.DP:
		TARGET_PATH = TARGET_PATH + "dp/"
		ATTACK_PATH = ATTACK_PATH + "dp/" + loss + "_" + optimizer + "/"
		target_model = get_model("CNN", num_classes[0])

	else:
		TARGET_PATH = TARGET_PATH + "basic/"
		ATTACK_PATH = ATTACK_PATH + "basic/"
		target_model = get_model(args.model, num_classes[0])

	if not os.path.exists(TARGET_PATH):
		os.makedirs(TARGET_PATH)
	
	if not os.path.exists(ATTACK_PATH):
		os.makedirs(ATTACK_PATH)
	
	

	################################################################################
	#                                                                              #
	#                            whole dataset mode                                #
	#                                                                              #
	################################################################################

	if args.mode == 0:
		attack_dataset = shadow_train

	################################################################################
	#                                                                              #
	#                            partial dataset mode                              #
	#                                                                              #
	################################################################################

	elif args.mode == 1:
		attack_dataset = target_train

	else:
		sys.exit("we have not supported this mode yet! 0c0")

	attack_length = int(0.8 * len(attack_dataset))

	attack_train, attack_test = torch.utils.data.random_split(attack_dataset, [attack_length, len(attack_dataset) - attack_length])
	# count_data(num_classes, attack_train)

	attack_trainloader = torch.utils.data.DataLoader(
		attack_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
	attack_testloader = torch.utils.data.DataLoader(
		attack_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

	result = []

	for epoch in checkpoint:
		print("++++++++++++++Attacking Epoch " + str(epoch) + " model++++++++++++++")
		# test_random_baseline(attack_testloader, args.dataset)
		if args.dataset == "UTKFace":
			dataset_type = "binary"
		elif args.dataset == "celeba":
			dataset_type = "macro"
		else:
			sys.exit("we have not supported this dataset yet! QwQ")

		acc_train, acc_test = train_attack_model(TARGET_PATH, ATTACK_PATH, num_classes[1], device, target_model, attack_trainloader, attack_testloader, epoch, loss, optimizer, dataset_type, args.mode, args.get_attack_set, args.round)
		result.append(acc_test)

	with open('./result/ol_' + str(args.round) + '.csv', 'a') as f:
		f.write(args.dataset + '_' + args.model + '_' + str(args.mode) + '_' + str(args.distill) + '_' + str(args.DP) + '_' + loss + '_' + optimizer)
		for test in result:
			for num in test:
				f.write(" " + str(round(num, 6)))
		f.write("\n")
