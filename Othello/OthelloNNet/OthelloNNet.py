import torch
import torch.nn as nn
import torch.nn.functional as F

# input turn as demension
class OthelloNNet(nn.Module):
    def __init__(self):
        super(OthelloNNet, self).__init__()
        self.board_x, self.board_y = 8, 8
        self.action_size = 64
        self.dropout = 0.2

        self.conv1 = nn.Conv2d(2, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear((16 * (self.board_x - 4) * (self.board_y - 4)), 256)
        self.fc_bn1 = nn.BatchNorm1d(256)

        self.policy_head1 = nn.Linear(256, 128)
        self.policy_head2 = nn.Linear(128, self.action_size)

        self.value_head1 = nn.Linear(256, 64)
        self.value_head2 = nn.Linear(64, 1)

    def forward(self, s):

        s = s.view(-1, 2, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn1(self.conv2(s)))
        s = F.relu(self.bn1(self.conv3(s)))
        s = F.relu(self.bn1(self.conv4(s)))
        s = s.view(-1, 16 * ((self.board_x - 4) * (self.board_y - 4)))

        s = F.relu(self.fc_bn1(self.fc1(s)))
        s = F.dropout(s, p=self.dropout, training=self.training)

        pi = F.relu(self.policy_head1(s))
        pi = F.relu(self.policy_head2(pi))
        pi = F.softmax(pi, dim=1)

        v = F.relu(self.value_head1(s))
        #print(2, v.data)
        v = torch .tanh(self.value_head2(v))
        #print(3, v.data)

        return pi, v

"""
Wouldn't we be training the model to make harsh value predictions
if we only train it to output 1 and -1 and not a probability?
"""
