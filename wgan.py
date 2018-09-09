"""
这里的gan选择wgan版本，即训练 Discriminator 的次数大于 Generator
"""
import numpy as np
import torch
import random
import math
import torch.nn as nn

from gru import RNN
from load_data import get_batch
from lib import MAE


class Generator(RNN):
	def __init__(self):
		super(Generator,self).__init__()

	def forward(self,x,h_state,mask,w1=0.75):
		# 1-w1 相当于正则项的系数，设置这个，有所提升
		r_out,h_state =  RNN.forward(self,x,h_state)
		# r_out = x*mask + r_out*(1-mask)
		r_out = (r_out*w1 + x*(1-w1))*mask + r_out*(1-mask)
		return r_out,h_state


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.rnn = nn.GRU(
			input_size  = DIMENSION,
			hidden_size = 256,
			num_layers  = 2,
			batch_first = True, # batch_size,time_step,dimension
		)
		self.out = nn.Sequential(
			nn.Linear(256,1),
			nn.Sigmoid(),
		)

	def forward(self,x):
		h_state = None
		r_out,h_state = self.rnn(x,h_state)
		# batch_size,time_step,66 -> batch_size,66 -> batch_size,1
		r_out = self.out(r_out[:,-1,:])
		return r_out


class WGAN:
	def __init__(self,discriminator,generator):
		discriminator.cuda()
		generator.cuda()
		self.D = discriminator
		self.G = generator

	def get_loss(self,valid_dataloader,rate=0.2):
		sum_loss = 0
		count = 0
		loss_func = MAE()
		for step,(b_x,b_y,mask) in enumerate(valid_dataloader):
			if random.random()>rate:
				continue
			b_x = b_x.cuda()
			b_y = b_y.cuda()
			mask = mask.cuda()
			prediction,_ = self.G(b_x,None,mask)
			loss = loss_func(prediction,b_y,mask)
			sum_loss += float(loss)
			count += 1
		maeLoss = sum_loss/count
		return maeLoss

	def reset_grad(self):
		self.G.zero_grad()
		self.D.zero_grad()

	def train(self,train_dataloader,valid_dataloader,EPOCH=10000):
		G_optim = torch.optim.Adam(self.G.parameters(),lr=1e-4)
		D_optim = torch.optim.Adam(self.D.parameters(),lr=1e-4)
		G_scheduler = torch.optim.lr_scheduler.StepLR(G_optim,10,gamma=0.25)
		D_scheduler = torch.optim.lr_scheduler.StepLR(D_optim,10,gamma=0.25)

		ones_label  = torch.ones(BATCH_SIZE,1).cuda()
		zeros_label = torch.zeros(BATCH_SIZE,1).cuda()

		for epoch in range(EPOCH):
			rate = (10-math.log(epoch+2))*10
			print (rate)
			G_scheduler.step()
			D_scheduler.step()

			prob1 = 0 
			prob2 = 0
			count1 = 0
			count2 = 0
			for step,(b_x,b_y,mask) in enumerate(train_dataloader):
				b_x = b_x.cuda()
				b_y = b_y.cuda()
				mask = mask.cuda()
				if step%100 < rate: 
					# 5/6 训练鉴别器
					# if step%60 == 0:
					# 	print ('\n========================D=========================') 
					prediction,_ = self.G(b_x,None,mask)
					D_real = self.D(b_y)
					D_fake = self.D(prediction)
					prob1 += torch.mean(D_real).item()*100 # 调试用
					count1 += 1 

					# 当D足够强大，那么D_real=1,D_fake=0,故D_loss=-1
					# D_loss = - (torch.mean(D_real) - torch.mean(D_fake))
					D_loss = torch.nn.functional.binary_cross_entropy(D_real, ones_label) + torch.nn.functional.binary_cross_entropy(D_fake, zeros_label)

					D_optim.zero_grad()
					D_loss.backward()
					D_optim.step()
					# if step%60 == 49:
					# 	print ('step %3d, Loss %.5f  |  %.2f%% %.2f%%'%(step,D_loss.item(),
					# 		torch.mean(D_real).item()*100,torch.mean(D_fake).item()))
					# for p in self.D.parameters():
						# p.data.clamp_(-0.01,0.01)
					
				else:
					# if step%60 == 50:
					# 	print ('\n========================G=========================') 
					# 1/6 训练生成器
					prediction,_ = self.G(b_x,None,mask)
					D_fake = self.D(prediction)
					prob2 += torch.mean(D_fake).item()*100 # 调试用
					count2 += 1 

					# 当G足够强大，那么D_fake=1,G_loss=-1
					# G_loss = - torch.mean(D_fake)
					G_loss = torch.nn.functional.binary_cross_entropy(D_fake, ones_label)

					G_optim.zero_grad()
					G_loss.backward()
					G_optim.step()
					# if step%60 == 59:
					# 	print ('step %3d, Loss %.5f  |  %.2f%%'%(step,G_loss.item(),torch.mean(D_fake).item()*100))
					

			maeloss1 = self.get_loss(train_dataloader,rate=0.1)
			maeloss2 = self.get_loss(valid_dataloader,rate=0.5)
			print ('>> EPOCH %3d | G_loss %.4f, D_loss %.4f | %.3f%% %.3f%% | maeloss %.5f %.5f'%(
				epoch,G_loss.item(),D_loss.item(),prob1/count1,prob2/count2,maeloss1,maeloss2))


if __name__ == "__main__":

	data_train = np.load('ts_data/data_train.npy')
	data_valid = np.load('ts_data/data_valid.npy')
	LR = 1e-4
	TIME_STEP  = 10
	BATCH_SIZE = 100
	DIMENSION  = 66
	missing_rate = 0.2
	
	train_dataloader = get_batch(data_train,TIME_STEP,BATCH_SIZE,DIMENSION,missing_rate)
	valid_dataloader = get_batch(data_valid,TIME_STEP,BATCH_SIZE,DIMENSION,missing_rate)

	## 训练网络
	# train(train_dataloader)
	G = Generator()
	D = Discriminator()
	wgan = WGAN(D,G)
	wgan.train(train_dataloader,valid_dataloader)

