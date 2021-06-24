import os
import sys
import torch
import pandas
import torchvision
torch.manual_seed(0)
import torch.nn as nn
import PIL.Image as Image
import torchvision.models as models
import torchvision.transforms as transforms


from tqdm import tqdm
from define_model import *
from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple

class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.files = os.listdir(root+'/UTKFace/processed/')
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []
        for txt_file in self.files:
            with open(self.root+'/UTKFace/processed/' + txt_file, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    if len(attrs) < 4 or int(attrs[2]) >= 4:
                        continue
                    self.lines.append(image_name+'jpg')


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        attrs = self.lines[index].split('_')

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(self.root+'/UTKFace/raw', self.lines[index]+'.chip.jpg').rstrip()

        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target

class CelebA(torch.utils.data.Dataset):
    base_folder = "celeba"

    def __init__(
            self,
            root: str,
            attr_list: str,
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.root = root
        self.transform = transform
        self.target_transform =target_transform
        self.attr_list = attr_list

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder, "img_celeba", self.filename[index]))

        target: Any = []
        for t, nums in zip(self.target_type, self.attr_list):
            if t == "attr":
                final_attr = 0
                for i in range(len(nums)):
                    final_attr += 2 ** i * self.attr[index][nums[i]]
                target.append(final_attr)
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

def prepare_dataset(model, dataset, attr):
    num_classes, dataset, target_model = get_model_dataset(model, dataset, attr=attr)
    train_length = int(len(dataset)*0.8)
    test_length = len(dataset) - int(len(dataset)*0.8)
    target_train, target_test= torch.utils.data.random_split(dataset, [train_length, test_length])

    return num_classes, target_train, target_test, target_model

def get_model_dataset(model_name, dataset_name, attr):
    root = './data/'
    if dataset_name == "UTKFace":
        if not isinstance(attr, list):
            sys.exit("please provide an attribute list")

        num_classes = []
        for a in attr:
            if a == "age":
                num_classes.append(117)
            elif a == "gender":
                num_classes.append(2)
            elif a == "race":
                num_classes.append(4)
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(a))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = UTKFaceDataset(root=root, attr=attr, transform=transform)

        if model_name == "alexnet":
            target_model = models.alexnet(num_classes=num_classes[0])

        elif model_name == "resnet18":
            target_model = models.resnet18(num_classes=num_classes[0])

        elif model_name == "vgg11":
            target_model = models.vgg11(num_classes=num_classes[0])

        elif model_name == "vgg19":
            target_model = models.vgg19(num_classes=num_classes[0])

        elif model_name == "CNN":
            target_model = CNN(num_classes=num_classes[0])
            
        else:
            sys.exit("we have not supported this model yet! :()")
        

    elif dataset_name == "celeba":
        if not isinstance(attr, list):
            sys.exit("please provide an attribute list")
        for a in attr:
            if a != "attr":
                raise ValueError("Target type \"{}\" is not recognized.".format(a))
            num_classes = [8, 4]
            # heavyMakeup MouthSlightlyOpen Smiling, Male Young
            attr_list = [[18, 21, 31], [20, 39]]

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = CelebA(root=root, attr_list=attr_list, target_type=attr, transform=transform)

        if model_name == "alexnet":
            target_model = models.alexnet(num_classes=num_classes[0])

        elif model_name == "resnet18":
            target_model = models.resnet18(num_classes=num_classes[0])

        elif model_name == "vgg11":
            target_model = models.vgg11(num_classes=num_classes[0])

        elif model_name == "vgg19":
            target_model = models.vgg19(num_classes=num_classes[0])

        elif model_name == "CNN":
            target_model = CNN(num_classes=num_classes[0])
            
        else:
            sys.exit("we have not supported this model yet! :()")


    return num_classes, dataset, target_model