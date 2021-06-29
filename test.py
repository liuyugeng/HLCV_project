from operator import mod
import torch
import torchvision.models as models

from utils import *
from train import *
from define_model import *

model = CNN(num_classes = 10)
print(model)
a = []
for name, _ in model.named_parameters():
    if "weight" in name:
        a.append("model." + name)
print(a)

x = torch.rand([1, 3, 64, 64])
def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

act = {}
layer = a[-2].split('.')
var = eval(layer[0] + '.' + layer[1])

var[int(layer[2])].register_forward_hook(get_activation('features', act))
output = model(x)
print(act['features'].flatten().shape[0])

