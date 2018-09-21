"""
pytorch 0.4.1

使用GRU进行时间序列修复

- 网路结构
	RNN(
	  (rnn): GRU(66, 256, num_layers=2, batch_first=True)
	  (out): Linear(in_features=256, out_features=66, bias=True)
	)

- 损失
	训练损失 MSE
	评价指标 MAE

- 学习器 Adam(_,lr=1e-3)
- 学习率 StepLR(20,gamma=0.3)
"""

import torch
import torch.nn as nn
import numpy as np

from load_data import get_batch
from lib import MSELoss_with_mask,MAE



class GRU(nn.Module):
	def __init__(self,input_size,hidden_size,output_size):
		"""
		NN、N1
		"""
		super(GRU,self).__init__()
		self.gru = nn.GRU(
			input_size  = input_size,
			hidden_size = hidden_size,
			num_layers  = 2,
			batch_first = True, 
		)
		self.out = nn.Sequential(
			nn.Linear(hidden_size,output_size),
		)	

	def forward(self,x):
		"""
		x 的形状为 (batch_size,time_step,dimension)
		"""
		h_state = None
		r_out,_ = self.gru(x,h_state)
		r_out   = self.out(r_out)
		return r_out


def get_loss(model,loader):
	"""
	计算 model 在 loader 上的损失，用于训练时的 early stop
	"""
	sum_loss = 0
	count = 0
	loss_func = MAE() # MSELoss_with_mask((1,0))
	for step,(b_x,b_y,mask,delta) in enumerate(loader):
		b_x  = b_x.cuda()
		b_y  = b_y.cuda()
		mask = mask.cuda()
		prediction = model(b_x)
		loss = loss_func(prediction,b_y,mask)
		sum_loss += float(loss)
		count += 1
	maeLoss = sum_loss/count
	return maeLoss


def train_Model(model,train_dataloader,valid_dataloader,
		patience=10,min_delta=0.0001,weight=(1,1)):

	optimizer = torch.optim.Adam(model.parameters(),lr=LR)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,20,gamma=0.3)
	loss_func = MSELoss_with_mask(weight)

	# 用于 early_stop 的变量
	min_loss_on_valid = 10000.
	patient_epoch = 0

	for epoch in range(200):
		scheduler.step()

		for step,(b_x,b_y,mask,delta) in enumerate(train_dataloader):
			b_x = b_x.cuda()
			b_y = b_y.cuda()
			mask = mask.cuda()

			prediction = model(b_x)
			loss = loss_func(prediction,b_y,mask)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		# 输出LOSS
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
	## 超参数
	TIME_STEP  = 10
	BATCH_SIZE = 100
	missing_rate = 0.2
	DIMENSION  = 66
	LR = 1e-3
	
	## 加载数据
	data_train_dir = 'ts_data/data_train.npy'
	data_valid_dir = 'ts_data/data_valid.npy'
	
	train_dataloader = get_batch(data_train_dir,None,
		TIME_STEP,BATCH_SIZE,DIMENSION,missing_rate,
		tmp_dir="data/tmp/train/")

	valid_dataloader = get_batch(data_valid_dir,None,
		TIME_STEP,BATCH_SIZE,DIMENSION,missing_rate,
		tmp_dir="data/tmp/train/")
	
	## 训练网络
	rnn = GRU(66,256,66)
	rnn.cuda()
	print(rnn)

	train_Model(rnn, train_dataloader, valid_dataloader,weight=(1,0))