from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import torch

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

train_loader = DataLoader(train_set, batch_size=50, shuffle=True, num_workers=0)

'''
X_val, X_test, y_val, y_test = train_test_split(
    test_set.data, test_set.labels, test_size=0.8, stratify=test_set.labels
)

X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_val = torch.LongTensor(y_val)
y_test = torch.LongTensor(y_test)
'''

# verify there are equal examples for each class
#unique, counts = np.unique(y_test, return_counts = True)
#print(dict(zip(unique, counts)))
# >>> {0: 400, 1: 400, 2: 400, 3: 400, 4: 400, 5: 400, 6: 400, 7: 400, 8: 400, 9: 400}

#unique, counts = np.unique(y_val, return_counts = True)
#print(dict(zip(unique, counts)))
# >>> {0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100, 8: 100, 9: 100}

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.transform:
            print(self.X[idx].shape)
            image = self.transform(self.X[idx])
            label = self.y[idx]
            return image, label
        else:
            return self.X[idx], self.y[idx]
'''
val_set = CustomDataset(X_val, y_val)
test_set = CustomDataset(X_test, y_test)
'''

val_loader = DataLoader(test_set, batch_size=50, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=50, shuffle=True, num_workers=0)
