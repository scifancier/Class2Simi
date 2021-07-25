import torch.nn as nn
import torch.nn.functional as F

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5,stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(400, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.T_revision = nn.Linear(2, 2, False)

    def forward(self, x, revision=False):
        correction = self.T_revision.weight
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        if revision == True:
            return out, correction
        else:
            return out
    

class LeNet(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz = 28):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz//4
        self.n_feat = 50 * feat_map_sz * feat_map_sz

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 500),
            nn.BatchNorm1d(500),
#            nn.ReLU(inplace=True),

        )
        self.last = nn.Linear(500, out_dim)  # Subject to be replaced dependent on task
        self.T_revision = nn.Linear(2, 2, False)


    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x, revision=False):
        correction = self.T_revision.weight
        x = self.features(x)
        out = self.logits(x)
        if revision == True:
            return out, correction
        else:
            return out



class LeNet_copy(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz = 28):
        super(LeNet_copy, self).__init__()
        feat_map_sz = img_sz//4
        self.n_feat = 50 * feat_map_sz * feat_map_sz

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 500),
            nn.BatchNorm1d(500),
#            nn.ReLU(inplace=True),

        )
        self.last = nn.Linear(500, out_dim)  # Subject to be replaced dependent on task


    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x



