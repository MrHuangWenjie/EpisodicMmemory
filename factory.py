import torch.nn as nn
import torch
import os


class ObjectClassifier(nn.Module):
    def __init__(self, name):
        super(ObjectClassifier, self).__init__()
        self.name = name
        self.con1 = torch.nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=(1, 1), stride=(1, 1))
        self.con2 = torch.nn.Conv2d(in_channels=8192, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        # self.con3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.con4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.con5 = torch.nn.Conv2d(in_channels=128, out_channels=8192, kernel_size=(1, 1), stride=(1, 1))
        self.con6 = torch.nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=(1, 1), stride=(1, 1))
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=8192)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=128)
        # self.batch_norm3 = torch.nn.BatchNorm2d(num_features=128)
        self.batch_norm4 = torch.nn.BatchNorm2d(num_features=128)
        self.batch_norm5 = torch.nn.BatchNorm2d(num_features=8192)
        self.batch_norm6 = torch.nn.BatchNorm2d(num_features=8192)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.batch_norm1(self.con1(x)))
        out = self.relu(self.batch_norm2(self.con2(out)))
        # out = self.relu(self.batch_norm3(self.con3(out)))
        out = self.relu(self.batch_norm4(self.con4(out)))
        out = self.relu(self.batch_norm5(self.con5(out)))
        out = self.sig(self.batch_norm6(self.con6(out)))
        return out


class ColorClassifier(nn.Module):
    def __init__(self, name):
        super(ColorClassifier, self).__init__()
        self.name = name
        self.con1 = torch.nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=(1, 1), stride=(1, 1))
        self.con2 = torch.nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=(1, 1), stride=(1, 1))
        self.con6 = torch.nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=(1, 1), stride=(1, 1))
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=8192)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=8192)
        self.batch_norm6 = torch.nn.BatchNorm2d(num_features=8192)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.batch_norm1(self.con1(x)))
        out = self.relu(self.batch_norm2(self.con2(out)))
        out = self.sig(self.batch_norm6(self.con6(out)))
        return out


class Classifier(nn.Module):
    def __init__(self, name):
        super(Classifier, self).__init__()
        self.name = name
        self.con1 = torch.nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=(1, 1), stride=(1, 1))
        self.con2 = torch.nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=(1, 1), stride=(1, 1))
        self.con6 = torch.nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=(1, 1), stride=(1, 1))
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=8192)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=8192)
        self.batch_norm6 = torch.nn.BatchNorm2d(num_features=8192)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.batch_norm1(self.con1(x)))
        out = self.relu(self.batch_norm2(self.con2(out)))
        out = self.sig(self.batch_norm6(self.con6(out)))
        return out


class Analyser(nn.Module):
    def __init__(self):
        super(Analyser, self).__init__()
        self.con1 = nn.Conv2d(in_channels=8192, out_channels=256, kernel_size=(1, 1), stride=(1, 1))
        self.con2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.con3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.con4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.con5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1))
        self.con6 = nn.Conv2d(in_channels=256, out_channels=8203, kernel_size=(1, 1), stride=(1, 1))
        # self.fc1 = nn.Linear(in_features=8192, out_features=8192)
        # self.fc2 = nn.Linear(in_features=8192, out_features=8192)
        # self.fc3 = nn.Linear(in_features=8192, out_features=8192)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.norm1 = nn.BatchNorm2d(num_features=256)
        self.norm2 = nn.BatchNorm2d(num_features=128)
        self.norm3 = nn.BatchNorm2d(num_features=128)
        self.norm4 = nn.BatchNorm2d(num_features=128)
        self.norm5 = nn.BatchNorm2d(num_features=256)
        self.norm6 = nn.BatchNorm2d(num_features=8203)

    def forward(self, x):
        out = self.relu(self.norm1(self.con1(x)))
        out = self.relu(self.norm2(self.con2(out)))
        out = self.relu(self.norm3(self.con3(out)))
        out = self.relu(self.norm4(self.con4(out)))
        out = self.relu(self.norm5(self.con5(out)))
        out = self.sig(self.norm6(self.con6(out)))
        # out = self.sig(self.fc(out.squeeze(dim=-1).squeeze(dim=-1))).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return out


class ControlAnalyser(nn.Module):
    def __init__(self):
        super(ControlAnalyser, self).__init__()
        self.con1 = nn.Conv2d(in_channels=8192, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        # self.con2 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), stride=(1, 1))
        self.con3 = nn.Conv2d(in_channels=128, out_channels=11, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.norm1 = nn.BatchNorm2d(num_features=128)
        # self.norm2 = nn.BatchNorm2d(num_features=1024)
        self.norm3 = nn.BatchNorm2d(num_features=11)

    def forward(self, x):
        out = self.relu(self.norm1(self.con1(x)))
        # out = self.relu(self.norm2(self.con2(out)))
        out = self.sig(self.norm3(self.con3(out)))
        return torch.reshape(out, [len(x), -1])


class MoveAnalyser(nn.Module):
    def __init__(self):
        super(MoveAnalyser, self).__init__()
        self.con1 = nn.Conv2d(in_channels=8192*2, out_channels=1024, kernel_size=(1, 1), stride=(1, 1))
        self.con2 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.con3 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.norm1 = nn.BatchNorm2d(num_features=1024)
        self.norm2 = nn.BatchNorm2d(num_features=128)
        self.norm3 = nn.BatchNorm2d(num_features=4)

    def forward(self, x):
        out = self.relu(self.norm1(self.con1(x)))
        out = self.relu(self.norm2(self.con2(out)))
        out = self.sig(self.norm3(self.con3(out)))
        return torch.reshape(out, [len(x), -1])