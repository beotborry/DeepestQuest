from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision import transforms
import numpy as np


# define transforms for dataset
# data augmentation 1 => random crop
# data augmentation 2 => horizontal flip
# reference for transforms code https://www.codegrepper.com/code-examples/python/torchvision+stl10+

train_transform = transforms.Compose([
    transforms.Resize([96, 96]),
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# change train set and test set
train_set = torchvision.datasets.STL10(
    root='./data',
    split='test',
    download=True,
    transform=train_transform
)

test_set = torchvision.datasets.STL10(
    root='./data',
    split='train',
    download=True,
    transform=test_transform
)

train_loader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)

val_set, test_set = random_split(test_set, [1000, 4000])

# verify there are equal examples for each class
#unique, counts = np.unique(val_set, return_counts = True)
#print(dict(zip(unique, counts)))
# >>> {0: 400, 1: 400, 2: 400, 3: 400, 4: 400, 5: 400, 6: 400, 7: 400, 8: 400, 9: 400}

#unique, counts = np.unique(y_val, return_counts = True)
#print(dict(zip(unique, counts)))
# >>> {0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100, 8: 100, 9: 100}

val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True, num_workers=0)

a = []
for X, y in val_loader:
    a.append(y.item())
