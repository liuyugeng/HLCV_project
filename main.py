import os
import sys
import torch
import argparse
import torch.multiprocessing
import torchvision.models as models

from utils import *
from train import *

def train_target(TARGET_PATH, device, target_model, train_loader, test_loader, use_DP, num_classes, noise, norm, batch_size):
    model = model_training(train_loader, test_loader, target_model, device, use_DP, num_classes, noise, norm, batch_size)

    for i in range(300):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("target training")
        acc_train = model.train()
        print("target testing")
        acc_test = model.test()

    filename = "target_model_" + str(noise) + "_" + str(norm) + ".pth"
    FILE_PATH = TARGET_PATH + filename
    model.saveModel(FILE_PATH)
    print("saved target model!!!\nFinished training!!!")

def train_distillation(MODEL_PATH, DL_PATH, device, target_model, student_model, train_loader, test_loader, noise, norm):
    MODEL_PATH = MODEL_PATH + "target_model_" + str(noise) + "_" + str(norm) + ".pth"
    # test_target_model(target_model, MODEL_PATH, test_loader, device)
    distillation = distillation_training(MODEL_PATH, train_loader, test_loader, student_model, target_model, device)

    for i in range(300):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("target distillation training")

        acc_distillation_train = distillation.train()
        print("target distillation testing")
        acc_distillation_test = distillation.test()
        
    result_path = DL_PATH + "target_model_" + str(noise) + "_" + str(norm) + ".pth"
    distillation.saveModel(result_path)
    print("saved distillation target model!!!\nFinished training!!!")

def test_target_model(model, PATH, dataset, device):
    model.load_state_dict(torch.load(PATH))
    model = model.to(device)
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for inputs, [targets, _] in dataset:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))


def train_attack_model(TARGET_PATH, ATTACK_PATH, output_classes, device, target_model, train_loader, test_loader, noise, norm, layer):
    TARGET_PATH = TARGET_PATH + "target_model_" + str(noise) + "_" + str(norm) + ".pth"
    ATTACK_PATH = ATTACK_PATH + "attack_model_" + str(-layer) +"pth"

    attack = attack_training(device, train_loader, test_loader, target_model, TARGET_PATH, ATTACK_PATH, layer)
    attack.init_attack_model(output_classes)

    for epoch in range(100):
        print("<======================= Epoch " + str(epoch+1) + " =======================>")
        print("attack training")
        acc_train = attack.train()
        print("attack testing")
        acc_test = attack.test()

    attack.saveModel(ATTACK_PATH)
    print("Saved Attack Model")
    print("Finished!!!")


    return acc_train, acc_test

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

    parser.add_argument('-g', '--gpu', type=str, default="1", 
                        help='Choose GPU')

    parser.add_argument('-d', '--dataset', type=str, default='celeba', 
                        help='Choose one Dataset, UTKFace or CelebA (default: UTKFace)')

    parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('-m', '--model', type=str, default="alexnet", 
                        help='Choose one model (default: alexnet)')

    parser.add_argument('-a', '--attribute', type=str, default=None, 
                        help='Choose one attribute for UTKFace(age, gender or race) and CelebA(landmarks, attr, identity or bbox, but we suggest using the attr to do membership inference attacks) (default: race)')

    parser.add_argument('-l', '--layer', type=int, default=-2,
                        help='choose the last layers (default: 1)')

    parser.add_argument('-dp', '--DP', type=str_to_bool, default=False,
                        help='if use differential privacy model, defualt no')

    parser.add_argument('-c', '--checkpoint', type=str_to_bool, default=True,
                        help='choose to save model in different epochs')

    parser.add_argument('-t', '--target', type=str_to_bool, default=False,
                        help='whether or not train target model')

    parser.add_argument('-dl', '--distill', type=str_to_bool, default=False,
                        help='where or not test a distillation model')

    parser.add_argument('-dt', '--distill_target', type=str_to_bool, default=False,
                        help='whether or not train distillation target model')

    parser.add_argument('-ne', '--noise', type=float, default=1.0,
                        help='choose noise for dp model')

    parser.add_argument('-nm', '--norm', type=float, default=1.0,
                        help='choose norm for dp model')

    parser.add_argument('-dpt', '--DP_type', type=int, default=1,
                        help='choose noise and norm for dp model')

    parser.add_argument('-da', '--delta', type=float, default=1e-5,
                        help='choose delta for dp model')


    args = parser.parse_args()

    attributes = args.attribute.split('_')
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.DP:
        if args.DP_type == 0:
            noise_set = {"UTKFace": 1.3, "celeba": 0.9}
            norm_set = {"UTKFace": 1.5, "celeba": 1.5}

        elif args.DP_type == 1:
            noise_set = {"UTKFace": 1.5, "celeba": 0.8}
            norm_set = {"UTKFace": 2.0, "celeba": 2.0}

        else:
            sys.exit("we have not supported this DP mode! hahaha")

        noise = noise_set[args.dataset]
        norm = norm_set[args.dataset]

    else:
        noise = None
        norm = None

    # get data set
    num_classes, target_train, target_test, target_model = prepare_dataset(args.model, args.dataset, attributes)

    # mkdir path
    TARGET_PATH = "./data/hlcv/target/" + args.dataset + "/" + args.model + "/"
    ATTACK_PATH = "./data/hlcv/attack/" + args.dataset + "/" + args.model + "/"

    if args.distill:
        DL_TARGET_PATH = TARGET_PATH + "distill/"
        DL_ATTACK_PATH = ATTACK_PATH + "distill/"

        if not os.path.exists(DL_TARGET_PATH):
            os.makedirs(DL_TARGET_PATH)
        
        if not os.path.exists(DL_ATTACK_PATH):
            os.makedirs(DL_ATTACK_PATH)
    
    if args.DP:
        TARGET_PATH = TARGET_PATH + "dp/"
        ATTACK_PATH = ATTACK_PATH + "dp/" + str(noise) + "_" + str(norm) + "/"
    else:
        TARGET_PATH = TARGET_PATH + "basic/"
        ATTACK_PATH = ATTACK_PATH + "basic/"

    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)
    
    if not os.path.exists(ATTACK_PATH):
        os.makedirs(ATTACK_PATH)

    target_trainloader = torch.utils.data.DataLoader(
        target_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    target_testloader = torch.utils.data.DataLoader(
        target_test, batch_size=args.batch_size, shuffle=True, num_workers=2)  

    # train target model
    if args.target:
        train_target(TARGET_PATH, device, target_model, target_trainloader, target_testloader, args.DP, num_classes, noise, norm, args.batch_size)
    
    #train distillation target model
    if args.distill:
        if args.model == "vgg19":
            student_target_model = models.vgg11(num_classes=num_classes[0])
        else:
            sys.exit("we have not supported this model for distillation yet! 0w0")
        if args.distill_target:
            train_distillation(TARGET_PATH, DL_TARGET_PATH, device, target_model, student_target_model, target_trainloader, target_testloader, noise, norm)
        target_model = student_target_model
        TARGET_PATH = DL_TARGET_PATH
    
    attack_length = int(0.5 * len(target_train))
    rest = len(target_train) - int(0.5 * len(target_train))

    attack_train, _ = torch.utils.data.random_split(target_train, [attack_length, rest])
    attack_test = target_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.dataset == "UTKFace":
        dataset_type = "binary"
    elif args.dataset == "celeba":
        dataset_type = "macro"
    else:
        sys.exit("we have not supported this dataset yet! QwQ")

    acc_train, acc_test = train_attack_model(TARGET_PATH, ATTACK_PATH, num_classes[1], device, target_model, attack_trainloader, attack_testloader, noise, norm, args.layer)
