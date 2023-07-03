import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class Data:
    def __init__(self, data_dir,train_batch_size,eval_batch_size,is_evaluate=False):
        pin_memory = True
        # if args.gpu is not None:
        #     pin_memory = True

        scale_size = 224

        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if not is_evaluate:
            trainset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Resize(scale_size),
                    transforms.ToTensor(),
                    normalize,
                ]))

            self.loader_train = DataLoader(
                trainset,
                batch_size=train_batch_size,
                shuffle=True,
                pin_memory=pin_memory,
                num_workers=8)

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]), )

        self.loader_test = DataLoader(
            testset,
            batch_size=eval_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8)
        
        
class Data_mini:
    def __init__(self, data_dir,train_batch_size,eval_batch_size,is_evaluate=False):
        pin_memory = True
        # if args.gpu is not None:
        #     pin_memory = True

        scale_size = 224

        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if not is_evaluate:
            trainset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(scale_size),
                    transforms.ToTensor(),
                    normalize,
                ]))

            self.loader_train = DataLoader(
                trainset,
                batch_size=train_batch_size,
                shuffle=True,
                pin_memory=pin_memory,
                num_workers=8)

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]), )

        self.loader_test = DataLoader(
            testset,
            batch_size=eval_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8)