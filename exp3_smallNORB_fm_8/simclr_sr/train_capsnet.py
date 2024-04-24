import os
import argparse
import torch 
import torchvision.transforms as transforms
import numpy as np
import neptune
from simclr import SimCLR
from simclr.modules import get_resnet 

from utils import yaml_config_hook
from torch.utils.data import  Dataset
from torchvision import transforms
from torch import nn, optim
# ref https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848/7
seed_value = 42
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `pytorch` pseudo-random generator at a fixed value
import torch
torch.manual_seed(seed_value)
 
# Capsule Network
from capsule_network import resnet20
from utilscapsnet import AverageMeter
from simclr.modules.identity import Identity
from norb import smallNORB
from torch.utils.data import Subset
def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           random_seed,
                           exp='azimuth',
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):

    data_dir = data_dir + '/' + dataset 
    from torchvision.transforms import InterpolationMode
    trans_train = [
                   
                transforms.Grayscale(num_output_channels=3), 
                transforms.Resize(129, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomCrop(113), 
                # transforms.RandomResizedCrop(32, scale=(0.4, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=32./255, contrast=0.3),
                transforms.ToTensor(),
            ]
    
    trans_valid = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(129, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(113),
            transforms.ToTensor(),
    ] 
    dataset = smallNORB(data_dir, train=True, download=True,
                    transform = None)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx = indices[split:]
    valid_idx = indices[:split]

    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)

    train_set_apply_transf = ApplyTransform(train_set, transform=transforms.Compose(trans_train))
    val_set_apply_transf = ApplyTransform(valid_set, transform=transforms.Compose(trans_valid))

    train_loader = torch.utils.data.DataLoader(
        train_set_apply_transf, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        val_set_apply_transf, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    print("type of train_loader", type(train_loader))
    print("type of valid_loader", type(valid_loader))
    print("Length of train_loader", len(train_loader))
    print("Length of valid_loader", len(valid_loader))
    
    return train_loader, valid_loader
 
def get_test_loader(data_dir,
                    dataset,
                    batch_size, 
                    num_workers=4,
                    pin_memory=False):

    data_dir = data_dir + '/' + dataset

    from torchvision.transforms import InterpolationMode
    if dataset == "smallNorb":
        trans = [
                   #  During test, we crop a 32 × 32 patch from the center of the image. Matrix capsules
            # with em routing
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(129, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(113),
            transforms.ToTensor(),
                 ]
        dataset = smallNORB(data_dir, train=False, download=True,
                                transform=transforms.Compose(trans))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    print("type of test_loader", type(data_loader))
    print("Length of test_loader", len(data_loader))
    return data_loader

class ApplyTransform(Dataset):
    # reference:https://stackoverflow.com/questions/56582246/correct-data-loading-splitting-and-augmentation-in-pytorch
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        if transform is None and target_transform is None:
            print("Transforms have failed")

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)
    
def save_checkpoint(state):
    epoch_num = state['epoch']
    filename = "simclr_linear_evaluation_after_pretrained" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints", filename)
    torch.save(state, ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': False}

    data_loader = get_train_valid_loader(
        "./data", "smallNorb", args.batch_size,
        2018, "full", 0.1,
        "True", **kwargs
    )
    train_loader = data_loader[0]
    valid_loader = data_loader[1]
    num_train = len(train_loader.dataset)
    num_valid = len(valid_loader.dataset)  
     
		
    print("Data loaders created successfully")
    print("Length of training dataset", len(train_loader.dataset))
    print("Length of validation dataset", len(valid_loader.dataset))
   
    # now we import testing dataset
    test_loader = get_test_loader("./data", "smallNorb", args.batch_size, **kwargs)
    num_test = len(test_loader.dataset)
    print("Length of testing dataset", num_test)
    
    capsule_network = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="SR").to(args.device)

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    encoder.avgpool = Identity() 
    encoder.fc = Identity()
    
    # PRETRAINED SIMCLR MODEL RESNET50 1x ON IMAGENET
    original_state_dict = torch.load("resnet50-1x.pth", map_location=args.device.type)

    # Since it is nested dictionary need to access the state_dict  
    nested_state_dict = original_state_dict['state_dict']

    original_keys = set(nested_state_dict.keys())
    modified_keys = set(encoder.state_dict().keys())
    filtered_state_dict = {k: v for k, v in nested_state_dict.items() if k in encoder.state_dict()}
    print("size of filtered_state_dict", len(filtered_state_dict))
    encoder.load_state_dict(filtered_state_dict, strict=True)
    excluded_keys = original_keys - set(filtered_state_dict.keys())

    # Step 4: Print excluded keys
    print("Excluded Keys:", excluded_keys)
    
    # initialize model
    model = SimCLR(encoder, capsule_network)
    model = model.to(args.device)
    # print(model) 
    print("Model loaded from pre-trained model successfully")
    
    for param in model.encoder.parameters():
        param.requires_grad = False
      
    for param in model.caps_net.parameters():
        param.requires_grad = True
    # check again if frozen
    model = model.to(args.device)
    
    start_epoch = 0
    criterion = nn.NLLLoss().to(args.device)
    optimizer = optim.SGD(model.caps_net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    num_train = len(train_loader.dataset)
    num_valid = len(valid_loader.dataset)
    num_test = len(test_loader.dataset)
    best_valid_acc = 0

    for epoch in range(args.logistic_epochs):
        losses = AverageMeter()
        accs = AverageMeter()
        encoder.eval()
        model.encoder.eval()
        model.caps_net.train()
        capsule_network.train()
        
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            # print("step: ", i)
            out = model(x)
            
            loss = criterion(out, y)

            # compute accuracy
            pred = torch.max(out, 1)[1]
            correct = (pred == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.data.item(), x.size()[0])
            accs.update(acc.data.item(), x.size()[0])

            # compute gradients and update SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
        train_loss, train_acc = losses.avg, accs.avg

        # run["after_pretraining/training/epoch/loss"].log(train_loss)
        # run["after_pretraining/training/epoch/acc"].log(train_acc)
        # evaluate on validation set
        with torch.no_grad():
            model.eval()

            losses = AverageMeter()
            accs = AverageMeter()

            for i, (x, y) in enumerate(valid_loader):
                x, y = x.to(args.device), y.to(args.device)

                out = model(x)
                
                loss = criterion(out, y)

                # compute accuracy
                pred = torch.max(out, 1)[1]
                correct = (pred == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.data.item(), x.size()[0])
                accs.update(acc.data.item(), x.size()[0])
        
        valid_loss, valid_acc = losses.avg, accs.avg
        
        # run["after_pretraining/validation/epoch/loss"].log(valid_loss)
        # run["after_pretraining/validation/epoch/acc"].log(valid_acc)  

        # decay lr
        scheduler.step()
        save_checkpoint(
                {'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'valid_acc': valid_acc
                    } 
                )
   
    correct = 0
    model.eval()

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(args.device), y.to(args.device)

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    perc = (100. * correct.data.item()) / (num_test)
    error = 100 - perc
    print(
        '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
            correct, num_test, perc, error)
    )
    
    # run["after_pretraining/testing/epoch/loss"].log(error)
    # run["after_pretraining/testing/epoch/acc"].log(perc)
    
    save_checkpoint(
        {'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'valid_acc': valid_acc
            } 
        )
    # run.stop()     