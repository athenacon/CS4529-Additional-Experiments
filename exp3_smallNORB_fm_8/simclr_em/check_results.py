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
from torch import optim
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
   
    # now we import testing dataset
    test_loader = get_test_loader("./data", "smallNorb", 8, **kwargs)
    num_test = len(test_loader.dataset)
    print("Length of testing dataset", num_test)
    
    capsule_network = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="EM").to(args.device)
    checkpoint = torch.load("check_results_checkpoints/simclr_linear_evaluation_after_pretrained_ckpt_epoch_2.pth.tar")

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    encoder.avgpool = Identity() 
    encoder.fc = Identity()
    
    # initialize model
    model = SimCLR(encoder, capsule_network)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(args.device)
    print(model) 
    print("Model loaded from pre-trained model successfully")
    model = model.to(args.device)
    
    start_epoch = 0

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
    