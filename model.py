import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, lr, epoch):
        super(MyModel, self).__init__()
        self.lr = lr
        self.epoch = epoch

        self.relu = nn.ReLU(inplace=False)

        self.conv3_64 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.conv64_64 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv64_128 = nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=False)
        self.conv128_128 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.conv128_256 = nn.Conv2d(128, 256, 3, padding=1, stride=2, bias=False)
        self.conv256_256 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.conv256_512 = nn.Conv2d(256, 512, 3, padding=1, stride=2, bias=False)
        self.conv512_512 = nn.Conv2d(512, 512, 3, padding=1, stride=1, bias=False)

        self.bn_64 = nn.BatchNorm2d(64)
        self.bn_128 = nn.BatchNorm2d(128)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_512 = nn.BatchNorm2d(512)

        self.downsample64_128 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.downsample128_256 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.downsample256_512 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.dropout_2 = nn.Dropout(0.2)
        self.dropout_3 = nn.Dropout(0.3)
        self.dropout_4 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(18432, 10)

    def forward(self, inputs):
        # 3 -> 64
        x = self.conv3_64(inputs)
        x = self.bn_64(x)


        # 64 -> 64
        _tmp = x
        x = self.conv64_64(x)
        x = self.bn_64(x)
        x = self.relu(x)
        x = self.conv64_64(x)
        x = self.bn_64(x)
        x = x + _tmp
        x = self.relu(x)
        x = self.dropout_2(x)

        # 64 -> 128
        _tmp = x
        x = self.conv64_128(x)
        x = self.bn_128(x)
        x = self.relu(x)
        x = self.conv128_128(x)
        x = self.bn_128(x)
        _tmp = self.downsample64_128(_tmp)
        x = x + _tmp
        x = self.relu(x)
        x = self.dropout_3(x)

        # 128 -> 128
        _tmp = x
        x = self.conv128_128(x)
        x = self.bn_128(x)
        x = self.relu(x)
        x = self.conv128_128(x)
        x = self.bn_128(x)
        _tmp = x + _tmp
        x = self.relu(x)
        x = self.dropout_4(x)

        x = F.avg_pool2d(x, 4)

        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)

        return x