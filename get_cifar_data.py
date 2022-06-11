import torchvision.datasets as datasets
import numpy as np
import torchvision.transforms as transforms
import os.path as osp
import pickle as pkl


mean_values = [0.5071,  0.4866,  0.4409]
std_values = [0.2009,  0.1984,  0.2023]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_values,
                            std=std_values),
])
num_classes = 10
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_values,
                            std=std_values),
])


train_datasets=datasets.CIFAR100(root='./data/cifar',train=True,
                        transform=transform_train,download=True)
test_datasets=datasets.CIFAR100(root='./data/cifar',train=False,
                        transform=transform_test,download=True)

all_images=np.concatenate((train_datasets.data,test_datasets.data))
all_lbl=train_datasets.targets+test_datasets.targets
all_data={}
all_data['data']=all_images
all_data['label']=all_lbl
SPLIT_PATH = 'data/cifar'
with open(osp.join(SPLIT_PATH, "all.pkl"),'wb') as g:
    pkl.dump(all_data,g)
print()