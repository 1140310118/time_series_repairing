import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as func

from gru import GRU
from load_data import get_batch
from lib import MAE


#==============判别器=================
# 10,66*2->1
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.rnn = nn.GRU(
			input_size  = 132,
			hidden_size = 256,
			num_layers  = 2,
			batch_first = True, # batch_size,time_step,dimension
		)
		self.out = nn.Sequential(
			nn.Linear(256,1),
			nn.Sigmoid(),
		)

	def forward(self,x,c):
		h_state = None
		input_  = torch.cat((x,c),dim=2)# (batch_size,time_step,-1)
		r_out,h_state = self.rnn(input_,h_state)
		r_out = self.out(r_out[:,-1,:])
		return r_out


#==============生成器=================
# 10,68->10,66
class Generator(nn.Module):
	def __init__(self,z_len):
		super(Generator,self).__init__()
		self.gru = nn.GRU(
			input_size  = 66+z_len,
			hidden_size = 256,
			num_layers  = 2,
			batch_first = True, 
		)
		self.out = nn.Sequential(
			nn.Linear(256,66),
		)

	def forward(self,z,c,mask):
		h_state = None
		if z is not None:
			input_  = torch.cat((z,c),dim=2)# (batch_size,time_step,-1)
		else:
			input_ = c
		r_out,_ = self.gru(input_,h_state)
		r_out   = self.out(r_out)
		r_out = c*mask+r_out*(1-mask)
		return r_out


#======================================

class GAN:
	def __init__(self,discriminator,generator):
		discriminator.cuda()
		generator.cuda()
		self.D = discriminator
		self.G = generator

	def zero_grad(self):
		self.D.zero_grad()
		self.G.zero_grad()

	def get_loss(self,valid_dataloader,z_len):
		sum_loss = 0
		count = 0
		loss_func = MAE()
		for step,(b_x,b_y,mask,delta) in enumerate(valid_dataloader):

			b_x = b_x.cuda()
			b_y = b_y.cuda()
			mask = mask.cuda()

			size1,size2 = b_x.size()[:2]
			prediction_sum = torch.zeros((10,size1,size2,66)).cuda()

			if z_len == 0:
				prediction = self.G(None,b_x,mask)
			else:
				for i in range(10):
					z = torch.randn((size1,size2,z_len)).cuda()
					prediction = self.G(z,b_x,mask)
					prediction_sum[i] = prediction
				prediction = torch.sum(prediction_sum,dim=0)/10

			loss = loss_func(prediction,b_y,mask)
			sum_loss += float(loss)
			count += 1

		maeLoss = sum_loss/count
		return maeLoss

	def train(self,train_dataloader,
		valid_dataloader,EPOCH=100,z_len=2):
		G_optim = torch.optim.Adam(self.G.parameters(),lr=LR)
		D_optim = torch.optim.Adam(self.D.parameters(),lr=LR)
		G_scheduler = torch.optim.lr_scheduler.StepLR(G_optim,5,gamma=0.5)
		D_scheduler = torch.optim.lr_scheduler.StepLR(D_optim,5,gamma=0.5)

		ones_label  = torch.ones(BATCH_SIZE,1).cuda()
		zeros_label = torch.zeros(BATCH_SIZE,1).cuda()
		d = 0
		for epoch in range(EPOCH):
			print ('EPOCH %3d'%epoch,end=" ")
			G_scheduler.step()
			D_scheduler.step()

			probG = probD = 0
			countG = countD = 1

			for step,(b_x,b_y,mask,delta) in enumerate(train_dataloader):
				b_x = b_x.cuda()
				b_y = b_y.cuda()
				mask = mask.cuda()
				
				if d<0.1:
					#======训练判别器=========
					size1,size2 = b_x.size()[:2]
					if z_len>0:
						z = torch.randn((size1,size2,z_len)).cuda()
					else:
						z = None
					prediction = G(z,b_x,mask)
					D_real = D(b_y,b_x)
					D_fake = D(prediction,b_x)
					probD += torch.sum(D_real+1-D_fake).item()
					countD += D_real.size(0)*2
					# D_loss_real = func.binary_cross_entropy(D_real, ones_label[:size1])
					# D_loss_fake = func.binary_cross_entropy(D_fake, zeros_label[:size1])

					# D_loss = D_loss_real + D_loss_fake
					D_loss = -(torch.mean(D_real)-torch.mean(D_fake))

					D_loss.backward(retain_graph=True)
					D_optim.step()

					for p in D.parameters():
						p.data.clamp_(-0.01, 0.01)

					self.zero_grad()

					d = torch.mean(D_real).item() - torch.mean(D_fake).item()
				else:
					#======训练生成器=========
					size1,size2 = b_x.size()[:2]
					if z_len>0:
						z = torch.randn((size1,size2,z_len)).cuda()
					else:
						z = None
					prediction = G(z,b_x,mask)
					D_fake = D(prediction,b_x)
					probG += torch.sum(D_fake).item()
					countG += D_fake.size(0)

					# G_loss = func.binary_cross_entropy(D_fake, ones_label[:size1])
					G_loss = -torch.mean(D_fake)

					G_loss.backward()
					G_optim.step()

					self.zero_grad()
					d = 0

			with torch.no_grad():
				probD /= countD
				probG /= countG
				# maeloss1 = self.get_loss(train_dataloader)
				maeloss2 = self.get_loss(valid_dataloader,z_len)
				print ('| G_prob %.5f, D_prob %.5f | %.5f'%(
					probG,probD,maeloss2))




if __name__ == "__main__":
	LR = 1e-5
	TIME_STEP  = 10
	BATCH_SIZE = 100
	DIMENSION  = 66
	missing_rate = 0.2
	Z_len = 66
	
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
	G = Generator(z_len=Z_len)
	D = Discriminator()
	gan = GAN(D,G)
	gan.train(train_dataloader,valid_dataloader,1000,z_len=Z_len)

