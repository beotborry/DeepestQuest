import torch
import torch.nn as nn
from tqdm import tqdm
from model import MyModel
from torch.optim import Adam
from dataloader import train_loader, val_loader, test_loader

def fit(model, epoch, optimizer, criterion, train_loader, val_loader, scheduler=None, gpu=-1):

    for _epoch in range(epoch):
        model.train()
        print("Epoch: {}".format(_epoch + 1))
        i = 0
        loss = 0
        for _X, _y in tqdm(train_loader):
            if gpu >= 0: _X, _y, model = _X.cuda(), _y.cuda(), model.cuda()
            y_pred = model(_X)

            _loss = criterion(y_pred, _y)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            i += 1
            loss += _loss

        print("Epoch: {}, Loss: {}".format(_epoch + 1, loss))
        print("Epoch: {}, Val Acc: {}, Val Loss: {}".format(_epoch + 1, evaluate(model, val_loader)))

def evaluate(model, data_loader, gpu=-1):
    model.eval()

    i = 0
    accuracy = 0
    loss = 0
    for _X, _y in data_loader:
        if gpu >= 0: _X, _y, model = _X.cuda(), _y.cuda(), model.cuda()
        y_pred = model(_X)
        loss += nn.CrossEntropyLoss()(y_pred, _y)
        y_pred = y_pred.argmax(dim=1)
        accuracy += sum(y_pred == _y) / len(_y)
        i += 1
    accuracy /= i
    loss /= i

    return accuracy, loss

def main():
    model = MyModel(
        epoch = 50,
        lr = 0.001
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    gpu = 1 if torch.cuda.is_available() else -1

    optimizer = Adam(model.parameters(), lr=model.lr)
    criterion = nn.CrossEntropyLoss()

    fit(model, epoch=model.epoch, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
        val_loader=val_loader, gpu=gpu)

    evaluate(model, test_loader)

if __name__ == '__main__':
    main()
    print("Done!")