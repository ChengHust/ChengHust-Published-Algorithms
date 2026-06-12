import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class ANN(nn.Module):
    def __init__(self, n=10, h=20, dropout=0.5):
        super(ANN, self).__init__()
        self.linear1 = nn.Linear(n, h, bias=True)
        self.bn1 = nn.BatchNorm1d(h)
        self.dp1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(h, 1, bias=True)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x
    

class ANNmodel(object):
    def __init__(self, n, h, dropout, batchsize, lr, epochs):
        self.n = n
        self.h = h
        self.lr = lr
        self.epochs = epochs
        self.dp_rate = dropout
        self.ANN = ANN(self.n, self.h, self.dp_rate)
        self.ANN.cuda()
        self.optimizer = optim.Adam(self.ANN.parameters(), lr)
        self.epochs = epochs
        self.batchsize = batchsize

    def train(self, pop_dec, label):
        self.ANN.train()
        n, d = pop_dec.shape
        indices = np.arange(n)
        criterion = nn.BCELoss()
        iter_no = (n + self.batchsize - 1) // self.batchsize

        for epoch in range(self.epochs):
            for iteration in range(iter_no):
                given_x = pop_dec[iteration * self.batchsize: (1 + iteration) * self.batchsize, :]
                given_y = label[iteration * self.batchsize: (1 + iteration) * self.batchsize]
                
                given_x = Variable(torch.from_numpy(given_x).cuda()).float()
                given_y = Variable(torch.from_numpy(given_y).cuda()).float()
                
                y_results = self.ANN(given_x)
                loss = criterion(y_results, given_y)
                self.ANN.zero_grad()
                loss.backward()
                self.optimizer.step()
            np.random.shuffle(indices)
            pop_dec = pop_dec[indices]
            label = label[indices]
    
    def predict(self, x):
        self.ANN.eval()
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, x.shape[1])
        x = Variable(torch.from_numpy(x).cuda()).float()
        with torch.no_grad():
            result = self.ANN.forward(x).cpu().data.numpy()
        return np.reshape(result, (batch_size, -1))
