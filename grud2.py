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
	def __init__(self,input_size,hidden_size,time_step=10):
		self.hidden_size = hidden_size
		self.time_step = time_step
		self.zl = nn.Linear(input_size*2+hidden_size,hidden_size)
		self.rl = nn.Linear(input_size*2+hidden_size,hidden_size)
		self.hl = nn.Linear(input_size*2+hidden_size,hidden_size)

		self.gamma_xl = nn.Linear(input_size,input_size)
		self.gamma_hl = nn.Linear(input_size,hidden_size)

	def forward(self,X,Mask,Delta,h_state=None):
		batch_size = X.size(0)
		if h_state is None:
			h_state = torch.zeros(batch_size,self.hidden_size).cuda()
		x_mean = torch.mean(X,dim=2)
		outs = []
		for i in range(self.time_step):
			h = self.step(X[:,i],h_state,Mask[:,i],Delta[:,i],x_mean)
			outs.append(h)
		outs = torch.stack(outs,dim=1)
		outs = self.out(outs)
		return outs,h

	def step(self,x,h,mask,delta,x_mean):
		delta_x = torch.exp(-torch.relu(self.gamma_xl(delta)))
		delta_h = torch.exp(-torch.relu(self.gamma_hl(delta)))

		x = mask*x + (1-mask)*(delta_x*x +(1-delta_x)*x_mean) # 缺失值的初步估计
		h = delta_h*h # h的旧值的估计值

		combined = torch.cat((x,h,mask),dim=1)
		r = torch.sigmoid(self.rl(combined))
		z = torch.sigmoid(self.zl(combined))

		combined2 = torch.cat((x,r*h,mask),dim=1)
		h_ = torch.tanh(self.hl(combined2))

		h = (1-z)*h + z*h_

		return h


class GRUD(nn.Module):
	def __init__(self,input_size,hidden_size,time_step=10):
		super(GRUD,self).__init__()

		self.hidden_size = hidden_size
		self.time_step = time_step

		self.zl = nn.Linear(input_size*2+hidden_size,hidden_size)
		self.rl = nn.Linear(input_size*2+hidden_size,hidden_size)
		self.hl = nn.Linear(input_size*2+hidden_size,hidden_size)

		self.gamma_xl = nn.Linear(input_size,input_size)
		self.gamma_hl = nn.Linear(input_size,hidden_size)

		self.gru = nn.GRU(
			input_size  = hidden_size,
			hidden_size = hidden_size*2,
			num_layers  = 2,
			batch_first = True,
			)

		self.out = nn.Sequential(
			# nn.Linear(hidden_size*2,hidden_size),
			# nn.ReLU(),
			nn.Linear(hidden_size,dimension))

	def step(self,x,h,mask,delta,x_mean):
		# delta_x = torch.exp(-torch.max(self.zeros1,self.gamma_xl(delta)))
		# delta_h = torch.exp(-torch.max(self.zeros2,self.gamma_hl(delta)))
		delta_x = torch.exp(-torch.relu(self.gamma_xl(delta)))
		delta_h = torch.exp(-torch.relu(self.gamma_hl(delta)))

		x = mask*x + (1-mask)*(delta_x*x +(1-delta_x)*x_mean) # 缺失值的初步估计
		h = delta_h*h # h的旧值的估计值

		combined = torch.cat((x,h,mask),dim=1)
		r = torch.sigmoid(self.rl(combined))
		z = torch.sigmoid(self.zl(combined))

		combined2 = torch.cat((x,r*h,mask),dim=1)
		h_ = torch.tanh(self.hl(combined2))

		h = (1-z)*h + z*h_

		return h

	def forward(self,X,Mask,Delta,h_state=None):
		"""
		X 的形状为 (batch_size,time_step,dimension)
		x h 的形状为 (batch_size,dimension)
		"""
		batch_size = X.size(0)
		if h_state is None:
			h_state = torch.zeros(batch_size,self.hidden_size).cuda()
		x_mean = torch.mean(X,dim=1) # 不是很确定

		outs = []
		for i in range(self.time_step):
			h = self.step(X[:,i],h_state,Mask[:,i],Delta[:,i],x_mean)
			outs.append(h)
		outs = torch.stack(outs,dim=1)
		# outs,_ = self.gru(outs,None)
		outs = self.out(outs)
		return outs,h

def get_loss(model,loader):
	sum_loss = 0
	count = 0
	# loss_func = MAE()
	loss_func = MSELoss_with_mask((0,1))
	for step,(b_x,b_y,mask,delta) in enumerate(loader):
		b_x  = b_x.cuda()
		b_y  = b_y.cuda()
		mask = mask.cuda()
		delta = delta.cuda()
		prediction,_ = model(b_x,mask,delta)
		loss = loss_func(prediction,b_y,mask)
		sum_loss += float(loss)
		count += 1
	maeLoss = sum_loss/count
	return maeLoss

def train_Model(model,train_dataloader,valid_dataloader,
	lr=1e-3,weight=(1,0.1),EPOCH=1000,patience=10,min_delta=0.0001):

	optimizer = torch.optim.Adam(model.parameters(),lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,20,gamma=0.5)
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
	data_train = np.load('ts_data/data_train.npy')[:30000]
	data_valid = np.load('ts_data/data_valid.npy')[:10000]
	
	TIME_STEP  = 10
	BATCH_SIZE = 100
	dimension  = 66
	missing_rate = 0.2
	
	train_dataloader = get_batch(data_train,TIME_STEP,BATCH_SIZE,dimension,missing_rate,has_delta=True)
	valid_dataloader = get_batch(data_valid,TIME_STEP,BATCH_SIZE,dimension,missing_rate,has_delta=True)

	rnn = GRUD(dimension,1024,time_step=10)
	rnn.cuda()
	print (rnn)

	train_Model(rnn,train_dataloader,valid_dataloader,weight=(1,0.05))
	