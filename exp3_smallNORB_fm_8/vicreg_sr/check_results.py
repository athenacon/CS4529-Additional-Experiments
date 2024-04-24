from pathlib import Path
import argparse 
import os
import random 
import neptune
from torch import nn, optim
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from utilscapsnet import AverageMeter
import resnet
from capsnet import resnet20 
from norb import smallNORB
from torch.utils.data import Subset
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
    filename = "vicreg_trained_on_imgnet" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints", filename)
    torch.save(state, ckpt_path)
           

from norb import smallNORB
from torch.utils.data import Subset

def get_test_loader(data_dir,
                    dataset,
                    batch_size, 
                    num_workers=4,
                    pin_memory=False):

    data_dir = data_dir + '/' + dataset

    from torchvision.transforms import InterpolationMode
    if dataset == "smallNorb":
        trans = [
                   #  During test, we crop a 32 Ã— 32 patch from the center of the image. Matrix capsules
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

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    parser.add_argument(
        "--train-percent",
        default=100,
        type=int,
        choices=(100, 10, 1),
        help="size of traing set in percent",
    )

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, metavar="N", help="mini-batch size"
    )
    
    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    main_worker(0, args)

def main_worker(gpu, args):
    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    backbone, _ = resnet.__dict__[args.arch](zero_init_residual=True)
    
    head = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="SR").to(device)
    model = nn.Sequential(backbone, head).to(device)
    checkpoint = torch.load("check_results_checkpoints/vicreg_trained_on_imgnet_ckpt_epoch_100.pth.tar")
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device)
    
    kwargs = {'num_workers': 8, 'pin_memory': False}

    test_loader = get_test_loader("./data", "smallNorb", args.batch_size, **kwargs)
    num_test = len(test_loader.dataset)
    print("Length of testing dataset", num_test)
    
    # test the model
    correct = 0
    model.eval()
    num_test = len(test_loader.dataset)
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

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
    
    # run.stop()     
if __name__ == "__main__":
    main()