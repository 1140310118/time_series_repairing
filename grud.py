"""
pytorch 0.4.1
"""

import torch
import copy
import numpy as np
import torch.nn as nn

from load_data import get_batch
from lib import MSELoss_with_mask,MAE


class GRUDCell(nn.Module):

	def __init__(self,input_size,hidden_size):
		super(GRUDCell,self).__init__()

		self.input_size  = input_size
		self.hidden_size = hidden_size

		self.zl = nn.Linear(input_size+hidden_size,hidden_size)
		self.rl = nn.Linear(input_size+hidden_size,hidden_size)
		self.hl = nn.Linear(input_size+hidden_size,hidden_size)
		
		self.gamma_xl = nn.Linear(input_size,input_size)
		self.gamma_hl = nn.Linear(input_size,hidden_size)

	def forward(self,x,mask,delta,x_mean,h):
		if h is None:
			batch_size = x.size(0)
			h = torch.zeros(batch_size,self.hidden_size).cuda()

		delta_h = torch.exp(-torch.relu(self.gamma_hl(delta)))
		delta_x = torch.exp(-torch.relu(self.gamma_xl(delta)))
		
		# x = mask*x + (1-mask)*(delta_x*x +(1-delta_x)*x_mean) # 缺失值的初步估计
		# h = delta_h*h # h的旧值的估计值

		# combined = torch.cat((x,h,mask),dim=1)
		combined = torch.cat((x,h),dim=1)
		r = torch.sigmoid(self.rl(combined))
		z = torch.sigmoid(self.zl(combined))

		# combined2 = torch.cat((x,r*h,mask),dim=1)
		combined2 = torch.cat((x,r*h),dim=1)
		h_ = torch.tanh(self.hl(combined2))

		h = (1-z)*h + z*h_

		return h

	def __repr__(self):
		return 'GRUD(in_features=%d, out_features=%d)'%(self.input_size,self.hidden_size)



class GRUD(nn.Module):
	def __init__(self,dimension,hidden_size,time_step=10):
		super(GRUD,self).__init__()

		self.dimension   = dimension
		self.hidden_size = hidden_size
		self.time_step   = time_step
		self.grud = GRUDCell(dimension,hidden_size)
		# self.grud2 = GRUDCell_t(dimension,hidden_size,time_step)
		# self.grud3 = GRUDCell_t(input_size,hidden_size,time_step)
		# self.grud4 = GRUDCell_t(input_size,hidden_size,time_step)

		self.out = nn.Sequential(
			nn.Linear(hidden_size,dimension))

	def forward(self,X,Mask,Delta,h1=None):
		"""
		X 的形状为 (batch_size,time_step,dimension)
		x h 的形状为 (batch_size,dimension)
		"""
		x_mean = torch.mean(X,dim=1)
		for i in range(self.time_step):
			h1 = self.grud(X[:,i],Mask[:,i],Delta[:,i],x_mean,h1)
		# x,h2 = self.grud2(X,Delta,h2)
		# x,h3 = self.grud3(X,Delta,h3)
		# x,h4 = self.grud3(X,Delta,h4)
		
		outs = self.out(h1).view(-1,1,self.dimension)
		return outs,None


def get_loss(model,loader):
	sum_loss = 0
	count = 0
	loss_func = nn.L1Loss()
	# loss_func = MSELoss_with_mask((0,1))
	for step,(b_x,b_y,mask,delta) in enumerate(loader):
		b_x  = b_x.cuda()
		b_y  = b_y.cuda()
		mask = mask.cuda()
		delta = delta.cuda()
		prediction,_ = model(b_x,mask,delta)
		loss = loss_func(prediction,b_y)
		sum_loss += float(loss)
		count += 1
	maeLoss = sum_loss/count
	return maeLoss

def train_Model(model,train_dataloader,valid_dataloader,
	lr=1e-3,weight=(1,0.1),EPOCH=1000,patience=10,min_delta=0.0001):

	optimizer = torch.optim.Adam(model.parameters(),lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,20,gamma=0.3)
	loss_func = MSELoss_with_mask(weight)

	min_loss_on_valid = 10000.
	patient_epoch = 0

	for epoch in range(EPOCH):
		scheduler.step()       
		for step,(b_x,b_y,mask,delta) in enumerate(train_dataloader):
			b_x = b_x.cuda()
			b_y = b_y.cuda()
			mask = mask.cuda()
			delta = delta.cuda()

			prediction,_ = model(b_x,mask,delta)
			loss = loss_func(prediction,b_y,mask)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		# early stop

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
	rnn = GRUD(DIMENSION,256)
	rnn.cuda()
	print(rnn)

	train_Model(rnn, train_dataloader, valid_dataloader,weight=(1,0))