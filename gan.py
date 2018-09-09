"""
这里的gan选择wgan版本，即训练 Discriminator 的次数大于 Generator
"""
import numpy as np
import random
import torch
import torch.nn as nn

from gru import RNN
from load_data import get_batch
from lib import MAE


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


class Generator(RNN):
	def __init__(self):
		super(Generator,self).__init__()

	def forward(self,x,h_state,mask):
		r_out,h_state =  RNN.forward(self,x,h_state)
		return r_out,h_state


class GAN:
	def __init__(self,discriminator,generator):
		discriminator.cuda()
		generator.cuda()
		self.D = discriminator
		self.G = generator

	def get_loss(self,valid_dataloader,rate=1):
		sum_loss = 0
		count = 0
		loss_func = MAE()
		for step,(b_x,b_y,mask) in enumerate(valid_dataloader):
			if random.random()>rate:
				continue
			b_x = b_x.cuda()
			b_y = b_y.cuda()
			mask = mask.cuda()
			prediction,_ = self.G(b_x,None)
			loss = loss_func(prediction,b_y,mask)
			sum_loss += float(loss)
			count += 1
		maeLoss = sum_loss/count
		return maeLoss

	def train(self,train_dataloader,valid_dataloader,EPOCH=100):
		G_optim = torch.optim.Adam(self.G.parameters(),lr=LR)
		D_optim = torch.optim.Adam(self.D.parameters(),lr=LR)
		G_scheduler = torch.optim.lr_scheduler.StepLR(G_optim,30,gamma=0.5)
		D_scheduler = torch.optim.lr_scheduler.StepLR(D_optim,30,gamma=0.5)

		ones_label  = torch.ones(BATCH_SIZE,1).cuda()
		zeros_label = torch.zeros(BATCH_SIZE,1).cuda()

		for epoch in range(EPOCH):
			G_scheduler.step()
			D_scheduler.step()

			prob1 = 0 
			prob2 = 0
			count = 0
			for step,(b_x,b_y,mask) in enumerate(train_dataloader):
				b_x = b_x.cuda()
				b_y = b_y.cuda()
				
				prediction,_ = G(b_x,None)
				D_real = D(b_y)
				D_fake = D(prediction)
				prob1 += torch.mean(D_real).item()*100
				prob2 += torch.mean(D_fake).item()*100
				count += 1

				D_loss = torch.nn.functional.binary_cross_entropy(D_real, ones_label) + torch.nn.functional.binary_cross_entropy(D_fake, zeros_label)
				G_loss = torch.nn.functional.binary_cross_entropy(D_fake, ones_label)

				D_optim.zero_grad()
				D_loss.backward(retain_graph=True)
				D_optim.step()

				G_optim.zero_grad()
				G_loss.backward()
				G_optim.step()

			maeloss1 = self.get_loss(train_dataloader,rate=0.1)
			maeloss2 = self.get_loss(valid_dataloader,rate=0.5)
			print ('EPOCH %3d | G_loss %.5f, D_loss %.5f | %.3f%% %.3f%% | maeloss %.5f %.5f'%(
				epoch,G_loss.item(),D_loss.item(),prob1/count,prob2/count,maeloss1,maeloss2))




if __name__ == "__main__":

	data_train = np.load('ts_data/data_train.npy')
	data_valid = np.load('ts_data/data_valid.npy')
	LR = 1e-5 
	TIME_STEP  = 10
	BATCH_SIZE = 100
	DIMENSION  = 66
	missing_rate = 0.2
	
	train_dataloader = get_batch(data_train,TIME_STEP,BATCH_SIZE,DIMENSION,missing_rate)
	valid_dataloader = get_batch(data_valid,TIME_STEP,BATCH_SIZE,DIMENSION,missing_rate)

	## 训练网络
	# train(train_dataloader)
	G = RNN()
	D = Discriminator()
	gan = GAN(D,G)
	gan.train(train_dataloader,valid_dataloader)

