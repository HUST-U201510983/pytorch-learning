# coding=gbk
import torch
import torch.nn as nn
from keras.datasets import mnist
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class MY_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, padding=1)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(in_features=7*7*50, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=100)
        self.linear3 = nn.Linear(in_features=100, out_features=10)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout2d = nn.Dropout2d(0.2)

    def forward(self, INPUT):
        RES = self.conv1(INPUT)
        RES = torch.relu(self.maxpooling1(RES))
        RES = torch.relu(self.maxpooling2(self.conv2(RES)))
        RES = self.dropout2d(RES)
        RES = RES.view(-1, 7*7*50)
        RES = self.linear1(RES)
        RES = torch.relu(RES)
        RES = self.dropout1(RES)
        RES = torch.relu(self.linear2(RES))
        RES = self.dropout2(RES)
        RES = self.linear3(RES)
        return torch.log_softmax(RES, dim=0)


class MNIST_DataSet(Dataset):
    def __init__(self, data_ndarray, label_ndarray, size=(224, 224)):
        self.files = data_ndarray
        self.labels = label_ndarray
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = np.array(Image.fromarray(self.files[item]).resize(self.size))
        label = self.labels[item]
        return img, label


data = mnist.load_data()
training_data = data[0][0]
training_label = data[0][1]
testing_data = data[1][0]
testing_label = data[1][1]


digitdset = MNIST_DataSet(training_data, training_label, size=(28, 28))
train_loader = DataLoader(digitdset, batch_size=32, shuffle=True, num_workers=2)
if __name__ == '__main__':
    batch = 0
    for imgs, labels in train_loader:
        pass
    #     batch += 1
    #     print(imgs.size())
    # print('一共有%d个集合' % batch)
