import torch.nn as nn


class smallCNN(nn.Module):
    def __init__(self, dropout_rate, n_classes):
        super(smallCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=1),
            nn.Conv2d(12, 24, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(nn.Linear(72 * 72 * 24, 100), nn.ReLU())
        self.classify = nn.Sequential(
            nn.Linear(100, n_classes), nn.LogSoftmax(dim=1)
        )  # TODO: remove
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.classify(out)
        return out


class simpleCNN(nn.Module):
    def __init__(self, dropout_rate, n_classes):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=1),
            nn.Conv2d(12, 12, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, stride=1),
            nn.Conv2d(24, 48, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(nn.Linear(34 * 34 * 48, 100), nn.ReLU())
        self.classify = nn.Sequential(
            nn.Linear(100, n_classes), nn.LogSoftmax(dim=1)
        )  # TODO: remove
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.classify(out)
        return out


class deepCNN(nn.Module):
    def __init__(self, dropout_rate, n_classes):
        super(deepCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=1),
            nn.Conv2d(12, 12, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, stride=1),
            nn.Conv2d(24, 24, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=1),
            nn.Conv2d(48, 48, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(nn.Linear(15 * 15 * 48, 100), nn.ReLU())
        self.classify = nn.Sequential(
            nn.Linear(100, n_classes), nn.LogSoftmax(dim=1)
        )  # TODO: remove
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.classify(out)
        return out
