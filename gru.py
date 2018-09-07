"""
pytorch 0.4.1
"""

import torch
import torch.nn as nn
import numpy as np

from load_data import get_batch


## 超参数
DIMENSION = 66
LR = 5e-4


class RNN(nn.Module):
	def __init__(self):
		super(RNN,self).__init__()
		self.rnn = nn.GRU(
			input_size  = DIMENSION,
			hidden_size = 256,
			num_layers  = 2,
			batch_first = True, # batch_size,time_step,dimension
			dropout     = 0.0,
		)
		# self.bn = nn.BatchNorm1d(256)
		self.out = nn.Linear(256,DIMENSION)

	def forward(self,x,h_state):
		r_out,h_state = self.rnn(x,h_state)
		# r_out = self.bn(r_out)
		return self.out(r_out),h_state


class MSELoss_with_mask(nn.Module):

	def __init__(self,weight=(1,1)):
		"""
		weight 全部与缺失值部分的比重
		"""
		super(MSELoss_with_mask,self).__init__()
		self.weight = weight
	
	def forward(self,y_predict,y_target,mask):
		
		one_num  = torch.sum(mask)
		zero_num = torch.sum(1-mask)
		mse  = nn.MSELoss()
		loss1 = mse(y_predict, y_target)
		loss2 = mse(y_predict*(1-mask),
					y_target *(1-mask)) *zero_num/(zero_num+one_num)
		w1,w2 = self.weight
		loss = (loss1*w1+loss2*w2)/(w1+w2)
		return loss


def get_loss(rnn,loader):
	sum_loss = 0
	count = 0
	loss_func = MSELoss_with_mask((0,1))
	for step,(b_x,b_y,mask) in enumerate(loader):
		b_x  = b_x.cuda()
		b_y  = b_y.cuda()
		mask = mask.cuda()
		prediction,_ = rnn(b_x,None)
		loss = loss_func(prediction,b_y,mask)
		sum_loss += float(loss)
		count += 1
	mseLoss = sum_loss/count
	return mseLoss


def train(rnn,loader,loader2):
	optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
	loss_func = MSELoss_with_mask((1,4))

	for epoch in range(250):
		for step,(b_x,b_y,mask) in enumerate(loader):
			b_x = b_x.cuda()
			b_y = b_y.cuda()
			mask = mask.cuda()

			prediction,_ = rnn(b_x,None)
			loss = loss_func(prediction,b_y,mask)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print ('%3d'%(epoch+1),"%.7f"%loss.item(),end=" ")
		print ("%.7f"%get_loss(rnn,loader2))


if __name__ == "__main__":
	data_train = np.load('ts_data/data_train.npy')
	data_valid = np.load('ts_data/data_valid.npy')
	TIME_STEP  = 10
	BATCH_SIZE = 100
	dimension  = 66
	missing_rate = 0.2
	
	train_dataloader = get_batch(data_train,TIME_STEP,BATCH_SIZE,dimension,missing_rate)
	valid_dataloader = get_batch(data_valid,TIME_STEP,BATCH_SIZE,dimension,missing_rate)

	rnn = RNN()
	rnn.cuda()
	print(rnn)

	train(rnn, train_dataloader, valid_dataloader)