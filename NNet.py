import time
import numpy as np
import os
import torch
from torch.nn import MSELoss, L1Loss
import torch.optim as optim
from Othello.OthelloNNet.OthelloNNet import OthelloNNet
from tqdm import tqdm


args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 32,
    'batch_count': 10,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
}


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NNet:
    def __init__(self, nnet):
        self.nnet = nnet
        self.board_x, self.board_y = 8, 8
        self.MSELoss = MSELoss()
        self.L1Loss = L1Loss()

        if args["cuda"]:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in tqdm(range(args["epochs"]), desc="Epochs"):
            self.nnet.train()

            batch_count = args["batch_count"]

            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=args["batch_size"])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards))
                target_pis = torch.FloatTensor(np.array(pis)).view(-1, 64)
                target_pis -= target_pis.min(1, keepdim=True)[0]
                target_pis /= target_pis.sum(1, keepdim=True)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).view(-1, 1)

                if args["cuda"]:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(out_pi, target_pis)
                l_v = self.loss_v(out_v, target_vs)
                #print(out_v.data[0], target_vs.data[0], l_v.data)
                #print(l_pi.data)
                total_loss = l_pi + l_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board, turn):
        """
        board: np array with board
        """
        tboard = np.full((8, 8), turn)
        board = np.stack((board, tboard))
        board = torch.FloatTensor(board)
        if args["cuda"]:
            board = board.contiguous().cuda()
        board = board.view(2, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        #print(pi.view(8, 8).data.cpu().numpy())

        return pi.view(8, 8).data.cpu().numpy(), v.view(1).data.cpu().numpy()

    def loss_pi(self, outputs, targets):
        """
        Type of cross entropy?
        normalize first
        """
        #return -torch.sum(targets * outputs) / targets.size()[0]
        return self.L1Loss(outputs, targets)

    def loss_v(self, outputs, targets):
        #return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        return self.L1Loss(outputs, targets)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args["cuda"] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
