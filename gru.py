"""
pytorch 0.4.1
"""

import torch
import torch.nn as nn
import numpy as np

from load_data import get_batch


## 超参数
DIMENSION = 66
LR = 1e-3


class RNN(nn.Module):
	def __init__(self):
		super(RNN,self).__init__()
		self.rnn = nn.GRU(
			input_size  = DIMENSION*2,
			hidden_size = 256,
			num_layers  = 2,
			batch_first = True,
			dropout     = 0.5,
		)
		self.out = nn.Linear(256,DIMENSION)

	def forward(self,x,h_state):
		r_out,h_state = self.rnn(x,h_state)
		return self.out(r_out),h_state


def get_loss(rnn,loader):
	sum_loss = 0
	count = 0
	loss_func = nn.MSELoss()
	for step,(b_x,b_y) in enumerate(loader):
		b_x = b_x.cuda()
		b_y = b_y.cuda()
		prediction,_ = rnn(b_x,None)
		loss = loss_func(prediction,b_y)
		sum_loss += loss
		count += 1
	mseLoss = sum_loss/count
	return mseLoss.item()


def train(rnn,loader,loader2):
	optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
	loss_func = nn.MSELoss()

	for epoch in range(100):
		for step,(b_x,b_y) in enumerate(loader):
			b_x = b_x.cuda()
			b_y = b_y.cuda()

			prediction,_ = rnn(b_x,None)
			loss = loss_func(prediction,b_y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (step+1)%100 == 0:
				print (epoch,step+1,"%.4f"%loss.item(),end=" ")
				print ("%.4f"%get_loss(rnn,loader2))


if __name__ == "__main__":
	data_train = np.load('ts_data/data_train.npy')
	data_valid = np.load('ts_data/data_valid.npy')[:10000]
	
	TIME_STEP  = 10
	BATCH_SIZE = 100
	dimension  = 66
	missing_rate = 0.2
	
	train_dataloader = get_batch(data_train,TIME_STEP,BATCH_SIZE,dimension,missing_rate)
	valid_dataloader = get_batch(data_valid,TIME_STEP,BATCH_SIZE,dimension,missing_rate)

	rnn = RNN()
	rnn.cuda()

	train(rnn, train_dataloader, valid_dataloader)