
from pathlib import Path
import argparse 
import os
import random 
import neptune
from torch import nn
from torchvision import transforms
import torch
from torch.utils.data import random_split, Dataset
import resnet
from capsnet import resnet20 
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
           

import scipy.io as sio
class affNIST(Dataset):
    # reference: https://github.com/fabio-deep/Variational-Capsule-Routing
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            image, label: sample data and respective label'''

    def __init__(self, data_path, shuffle=False, transform=None):

        self.data_path = data_path
        self.shuffle = shuffle
        self.transform = transform
        self.split = self.data_path.split('/')[-1]

        if self.split == 'train':
            for i, file in enumerate(os.listdir(data_path)):
                # load dataset .mat file batch
                self.dataset = sio.loadmat(os.path.join(data_path, file))
                # concatenate the 32 .mat files to make full dataset
                if i == 0:
                    self.data = np.array(self.dataset['affNISTdata']['image'][0][0])
                    self.labels = np.array(self.dataset['affNISTdata']['label_int'][0][0])
                else:
                    self.data = np.concatenate((self.data,
                        np.array(self.dataset['affNISTdata']['image'][0][0])), axis=1)
                    self.labels = np.concatenate((self.labels,
                        np.array(self.dataset['affNISTdata']['label_int'][0][0])), axis=1)

            # (N, 1, 40, 40) <- (1, 40, 40, N) <- (40*40, N)
            self.data = np.moveaxis(self.data.reshape(1,40,40,-1), -1, 0)
            # (N,)
            # labels are 2D, squeeze to 1D
            self.labels = self.labels.squeeze()
        else:
            print("yes")
            # load valid/test dataset .mat file
            self.dataset = sio.loadmat(os.path.join(self.data_path, self.split+'.mat'))
            # (40*40, N)
            self.data = np.array(self.dataset['affNISTdata']['image'][0][0])
            # (N, 1, 40, 40) <- (1, 40, 40, N) <- (40*40, N)
            self.data = np.moveaxis(self.data.reshape(1,40,40,-1), -1, 0)
            # (N,)
            self.labels = np.array(self.dataset['affNISTdata']['label_int'][0][0]).squeeze()

        self.data = self.data.squeeze()

        if self.shuffle: # shuffle the dataset
            idx = np.random.permutation(self.data.shape[0])
            self.data = self.data[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        # Returns the total number of samples in the dataset,
        # which is simply the first dimension of the data array 
        return self.data.shape[0]

    def __getitem__(self, idx):
        #  Fetches the idx-th sample from the dataset.
        # It applies any specified transformations to the image data 
        # before returning it along with its corresponding label.

        image = self.data[idx]

        if self.transform is not None:
            image = self.transform(image) 
        return image, self.labels[idx] # (X, Y)



def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           random_seed,
                           exp='azimuth',
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=False):

    data_dir = data_dir +  '/' + dataset 
    from torchvision.transforms import InterpolationMode
    transf = {'train': transforms.Compose([
                
                transforms.Pad(6),
                
                transforms.Grayscale(num_output_channels=3), 
                transforms.Resize(113, interpolation=InterpolationMode.BICUBIC),

                
                transforms.RandomAffine(
                    degrees=0, # No rotation
                    translate=(0.2, 0.2), # Translate up to 20% of the image size per the attention paper
                    scale=None, # Keep original scale
                    shear=None, # No shear
                    interpolation=InterpolationMode.NEAREST, # Nearest neighbor interpolation
                    fill=0 # Fill with black color for areas outside the image
                ),
                transforms.ToTensor(),
                # transforms.Normalize((0.13066047,), (0.30810780,))
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                ]),
              
        'test':  transforms.Compose([ 
                # transforms.RandomCrop((28, 28), padding=6),
                transforms.Pad(6),
                transforms.Resize(113, interpolation=InterpolationMode.BICUBIC),
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToTensor(),
                # transforms.Normalize((0.13066047,), (0.30810780,))
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                ])}
    
    dataset_paths ={ 'train': data_dir + '/train', 'test': data_dir + '/test'}
    config = {'train': True, 'test': False}
    import torchvision
    datasets = {i: torchvision.datasets.MNIST(root=dataset_paths[i], transform=transf[i],
        train=config[i], download=True) for i in config.keys()}
    
    # split train into train and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].targets,
        n_classes=10,
        n_samples_per_class=np.repeat(500, 10)) # 500 per class

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Pad(6),
                         
         transforms.Grayscale(num_output_channels=3), 

            # transforms.Pad(6),
            
        transforms.Resize(113, interpolation=InterpolationMode.BICUBIC),

                transforms.RandomAffine(
                    degrees=0, # No rotation
                    translate=(0.2, 0.2), # Translate up to 20% of the image size per the attention paper
                    scale=None, # Keep original scale
                    shear=None, # No shear
                    interpolation=InterpolationMode.NEAREST, # Nearest neighbor interpolation
                    fill=0 # Fill with black color for areas outside the image
                ),
            transforms.ToTensor(),
            # transforms.Normalize((0.13066047,), (0.30810780,))
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
                transforms.ToPILImage(),
         transforms.Grayscale(num_output_channels=3), 
                # transforms.RandomCrop((28, 28), padding=6),
                # transforms.Pad(6),
                
                transforms.Resize(113, interpolation=InterpolationMode.BICUBIC),

                transforms.ToTensor(),
                # transforms.Normalize((0.13066047,), (0.30810780,))
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                ])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
        labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
        labels=labels['valid'], transform=transf['valid'])

    config = {'train': True, 'train_valid': True,
        'valid': False, 'test': False}
    from torch.utils.data import DataLoader
    dataloaders = {i: DataLoader(datasets[i], num_workers=8, pin_memory=True,
        batch_size=batch_size, shuffle=config[i]) for i in config.keys()}
   
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    return train_loader, valid_loader
 
class CustomDataset(Dataset):
    """ Creates a custom pytorch dataset, mainly
        used for creating validation set splits. """
    def __init__(self, data, labels, transform=None):
        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])
        if isinstance(data, torch.Tensor):
            data = data.numpy() # to work with `ToPILImage'
        self.data = data[idx]
        self.labels = labels[idx]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.data[idx])
        return image, self.labels[idx]
def random_split(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set. """
    train_X, train_Y, valid_X, valid_Y = [],[],[],[]

    for c in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == c).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[c], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_X.extend(data[train_samples])
        train_Y.extend(labels[train_samples])
        valid_X.extend(data[valid_samples])
        valid_Y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'train': torch.stack(train_X), 'valid': torch.stack(valid_X)}, \
            {'train': torch.stack(train_Y), 'valid': torch.stack(valid_Y)}
    else:
        # transforms list of np arrays to tensor
        return {'train': torch.from_numpy(np.stack(train_X)), \
                'valid': torch.from_numpy(np.stack(valid_X))}, \
            {'train': torch.from_numpy(np.stack(train_Y)), \
             'valid': torch.from_numpy(np.stack(valid_Y))}




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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    backbone, _ = resnet.__dict__[args.arch](zero_init_residual=True)
    
    head = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="EM").to(device)
    
    
    model = nn.Sequential(backbone, head).to(device)
    # load 
    model.load_state_dict(torch.load('check_results_checkpoints/vicreg_trained_on_imgnet_ckpt_epoch_100.pth.tar')['model_state'])
    
    # load data
         
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    
    from torchvision.transforms import InterpolationMode
    # test on padded mnist after each epoch until accuracy is >92.2 and <92.3
    transform_MNIST = transforms.Compose(      
    [   transforms.Grayscale(num_output_channels=3), 
        # transforms.Pad(6),
        transforms.Resize(113, interpolation=InterpolationMode.BICUBIC),

        transforms.ToTensor(),
        normalize   ]
    )
    from torchvision import datasets
    test_MNIST_dataset = datasets.MNIST(
        'data/',
        train=False,  # Load the test set
        download=True,
        transform=transform_MNIST,
    ) 
    
    # loader for test_MNIST_dataset
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    ) 
    test_MNIST_loader = torch.utils.data.DataLoader(test_MNIST_dataset, shuffle=False, **kwargs)
    
    # then finally test on affnist.
    # affNIST dataset 
    transform_affNIST = transforms.Compose(
[    transforms.ToPILImage(),

    transforms.Grayscale(num_output_channels=3), 
         transforms.Resize(113, interpolation=InterpolationMode.BICUBIC),

        transforms.ToTensor(),
        normalize
    ])

    # we only need testing dataset of affNIST
    test_dataset_path = 'affNist_transformed/test'
    test_dataset = affNIST(data_path=test_dataset_path, transform=transform_affNIST)
    print("Testing dataset loaded successfully")  
    
    # affNIST
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )  
    test_AffNIST_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **kwargs)
    
    print("Data loaders created successfully") 
    print("Length of testing on padded mnist: ", len(test_MNIST_loader.dataset))
    print("Length of testing on affnist: ", len(test_AffNIST_loader.dataset))
      
    num_test_mnist = len(test_MNIST_loader.dataset)
    num_test_affnist = len(test_AffNIST_loader.dataset)
    correct = 0
    model.eval()

    for i, (x, y) in enumerate(test_MNIST_loader):
        x, y = x.to(device), y.to(device)

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100. * correct.data.item()) / (num_test_mnist)
    error = 100 - perc
    print(
        '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
            correct, num_test_mnist, perc, error)
    )
     
    # we now test on affnist  
    correct = 0
    model.eval()

    for i, (x, y) in enumerate(test_AffNIST_loader):
        x, y = x.to(device), y.to(device)

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    perc = (100. * correct.data.item()) / (num_test_affnist)
    error = 100 - perc
    print(
        '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
            correct, num_test_affnist, perc, error)
    ) 
        
if __name__ == "__main__":
    main()