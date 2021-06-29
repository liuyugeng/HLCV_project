import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils import *
from opacus import PrivacyEngine
from torch.optim import lr_scheduler
from opacus.utils import module_modification
from opacus.dp_model_inspector import DPModelInspector
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

class model_training():
    def __init__(self, trainloader, testloader, model, device, use_DP, num_classes, noise, norm, batch_size):
        self.use_DP = use_DP
        self.device = device
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        self.noise_multiplier, self.max_grad_norm = noise, norm
        
        if self.use_DP:
            self.net = module_modification.convert_batchnorm_modules(self.net)
            inspector = DPModelInspector()
            inspector.validate(self.net)
            privacy_engine = PrivacyEngine(
                self.net,
                batch_size=batch_size,
                sample_size=len(self.trainloader.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                secure_rng=False,
            )
            print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
            privacy_engine.attach(self.optimizer)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

    # Training
    def train(self):
        self.net.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, [targets, _]) in enumerate(self.trainloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if self.use_DP:
            epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B1: %.3f \u03B5: %.3f \u03B4: 1e-5" % (best_alpha, epsilon))
                
        self.scheduler.step()

        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total


    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, [targets, _] in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total


class distillation_training():
    def __init__(self, PATH, trainloader, testloader, model, teacher, device):
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        self.PATH = PATH
        self.teacher = teacher.to(self.device)
        self.teacher.load_state_dict(torch.load(self.PATH))
        self.teacher.eval()

        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

    def distillation_loss(self, y, labels, teacher_scores, T, alpha):
        loss = self.criterion(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1))
        loss = loss * (T*T * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
        return loss

    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, [targets, _]) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            teacher_output = self.teacher(inputs)
            teacher_output = teacher_output.detach()
    
            loss = self.distillation_loss(outputs, targets, teacher_output, T=2.0, alpha=0.95)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

        self.scheduler.step()
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total

    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, [targets, _] in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total

class attack_training():
    def __init__(self, device, attack_trainloader, attack_testloader, target_model, TARGET_PATH, ATTACK_PATH, layer):
        self.device = device
        self.TARGET_PATH = TARGET_PATH
        self.ATTACK_PATH = ATTACK_PATH
        self.layer = layer

        self.target_model = target_model.to(self.device)
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        self.target_model.eval()

        self.attack_model = None

        self.attack_trainloader = attack_trainloader
        self.attack_testloader = attack_testloader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

    def _get_activation(self, name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    def init_attack_model(self, output_classes):
        x = torch.rand([1, 3, 64, 64]).to(self.device)
        input_classes = self.get_middle_output(x).flatten().shape[0]
        self.attack_model = get_attack_model(inputs_classes=input_classes, outputs_classes=output_classes)
        self.attack_model.to(self.device)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-3)

    def get_middle_output(self, x):
        temp = []
        for name, _ in self.target_model.named_parameters():
            if "weight" in name:
                temp.append(name)

        if -self.layer > len(temp):
            raise IndexError('layer is out of range')

        name = temp[self.layer].split('.')
        var = eval('self.target_model.' + name[0])
        out = {}
        var[int(name[1])].register_forward_hook(self._get_activation(str(self.layer), out))
        _ = self.target_model(x)

        return out[str(self.layer)]

    # Training
    def train(self):
        self.attack_model.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, [_, targets]) in enumerate(self.attack_trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            oracles = self.get_middle_output(inputs)
            outputs = self.attack_model(oracles)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # self.scheduler.step()

        final_result = 1.*correct/total
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return final_result

    def test(self):
        self.attack_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, [_, targets] in self.attack_testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                oracles = self.get_middle_output(inputs)
                outputs = self.attack_model(oracles)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        final_result = 1.*correct/total
        print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)