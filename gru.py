"""
pytorch 0.4.1

使用GRU进行时间序列修复

- 网路结构
	RNN(
	  (rnn): GRU(66, 256, num_layers=2, batch_first=True)
	  (out): Linear(in_features=256, out_features=66, bias=True)
	)

- 损失
	训练损失 MSE_with_mask 1:3/5/5
	评价指标 MAE

- 学习器 Adam(_,lr=1e-3)
- 学习率 StepLR(30,gamma=0.5)
"""

import torch
import torch.nn as nn
import numpy as np

from load_data import get_batch
from lib import MSELoss_with_mask,MAE



class RNN(nn.Module):
	def __init__(self,dimension,hidden_size=512):
		"""
		dimension:待修复的时间序列的维度 
		"""
		super(RNN,self).__init__()
		self.rnn = nn.GRU(
			input_size  = dimension,
			hidden_size = hidden_size,
			num_layers  = 2,
			batch_first = True, # batch_size,time_step,dimension
		)
		self.out = nn.Sequential(
			# nn.ReLU(),
			nn.Linear(hidden_size,dimension),
			# nn.Sigmoid()
		)
		

	def forward(self,x,h_state):
		r_out,h_state = self.rnn(x,h_state)
		r_out = self.out(r_out)
		return r_out,h_state


def get_loss(model,loader):
	sum_loss = 0
	count = 0
	# loss_func = MAE()
	loss_func = MSELoss_with_mask((0,1))
	for step,(b_x,b_y,mask) in enumerate(loader):
		b_x  = b_x.cuda()
		b_y  = b_y.cuda()
		mask = mask.cuda()
		prediction,_ = model(b_x,None)
		loss = loss_func(prediction,b_y,mask)
		sum_loss += float(loss)
		count += 1
	maeLoss = sum_loss/count
	return maeLoss


def train_Model(model,train_dataloader,valid_dataloader,
		patience=10,min_delta=0.00001,weight=(1,1)):
	optimizer = torch.optim.Adam(model.parameters(),lr=LR)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,30,gamma=0.5)
	loss_func = MSELoss_with_mask(weight)

	min_loss_on_valid = 10000.
	patient_epoch = 0

	for epoch in range(200):
		scheduler.step()

		for step,(b_x,b_y,mask) in enumerate(train_dataloader):
			b_x = b_x.cuda()
			b_y = b_y.cuda()
			mask = mask.cuda()

			prediction,_ = model(b_x,None)
			loss = loss_func(prediction,b_y,mask)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			print ('%3d'%(epoch+1),"%.7f"%loss.item(),end=' ')
			print ("%.7f"%get_loss(model,valid_dataloader),patient_epoch)
			valid_loss = get_loss(model,valid_dataloader)
			if min_loss_on_valid - valid_loss > min_delta:
				patient_epoch = 0
				min_loss_on_valid = valid_loss
			else:
				patient_epoch += 1
				if patient_epoch >= patience:
					print ('Early Stopped at Epoch:',epoch)
					break





if __name__ == "__main__":
	## 加载数据
	data_train = np.load('ts_data/data_train.npy')#[:30000]
	data_valid = np.load('ts_data/data_valid.npy')#[:10000]
	TIME_STEP  = 10
	BATCH_SIZE = 100
	missing_rate = 0.2
	DIMENSION  = 66
	LR = 1e-3
	
	train_dataloader = get_batch(data_train,TIME_STEP,BATCH_SIZE,DIMENSION,missing_rate)
	valid_dataloader = get_batch(data_valid,TIME_STEP,BATCH_SIZE,DIMENSION,missing_rate)
	valid_dataloader2 = get_batch(data_valid,TIME_STEP,1,DIMENSION,missing_rate,False)

	## 训练网络
	rnn = RNN(DIMENSION,256)
	rnn.cuda()
	print(rnn)

	train_Model(rnn, train_dataloader, valid_dataloader,weight=(1,0.05))