"""
pytorch 0.4.1
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import copy



# 全局参数的设置
TIME_STEP  = 10
TS_SIZE = 66 # 时间序列的宽度
LR = 0.01

# 加载数据
data_train = np.load('ts_data/data_train.npy')
data_valid = np.load('ts_data/data_valid.npy')

def make_data(data,missing_rate=0.2):
	"""缺失值直接赋值为0，这个地方有待商榷"""
	data = data.reshape(-1,TIME_STEP,TS_SIZE)
	Y = data
	ts_missing = copy.deepcopy(data) # 带有缺失值的mask
	mask = (np.random.random(X.shape) > missing_rate).astype(int) 
	X = 
	X,Y = torch.FloatTensor(X),torch.FloatTensor(Y)
	torch_dataset = Data.TensorDataset(X,Y)
	return torch_dataset

train_loader = Data.DataLoader(
	dataset=make_data(data_train[:1000]),
	batch_size=TIME_STEP,
	shuffle=False,
	num_workers=1,
)

valid_loader = Data.DataLoader(
	dataset=make_data(data_valid[:1000]),
	batch_size=TIME_STEP,
	shuffle=False,
	num_workers=1,
)

# GRU 网络结构

class GRU(nn.Module):
	def __init__(self):
		super(GRU,self).__init__()
		self.gru = nn.GRU(
			input_size  = INPUT_SIZE,
			hidden_size = 256,
			num_layers  = 3,
			batch_first = True, # (batch_size,time_size,66)
			dropout     = 0.5,
		)
		self.out = nn.Sequential(
			nn.Linear(66,128),
			# nn.BatchNorm1d(128),
			nn.Dropout(0.5),
			nn.ReLU(),
			nn.Linear(128,66),
			# nn.BatchNorm1d(66),
			nn.Sigmoid(),
			)

	def forward(self,x,h_state):
		# r_out, h_state = self.gru(x,h_state)
		r_out = x
		r_out = self.out(r_out)
		return r_out,h_state


gru = GRU()
gru.cuda()
# print (gru)

# 训练网络
def get_loss(loader):
	sum_loss = 0
	h_state = None
	count = 0
	loss_func = nn.MSELoss()
	for step,(b_x,b_y) in enumerate(loader):
		b_x = b_y.view(1,TIME_STEP,-1).cuda()
		b_y = b_y.view(1,TIME_STEP,-1).cuda()
		prediction,h_state = gru(b_x,h_state)
		# h_state = h_state.data # ?
		loss = loss_func(prediction,b_y)
		sum_loss += loss
		count += 1
	mseLoss = sum_loss/count
	return mseLoss.item()


def train():
	optimizer = torch.optim.Adam(gru.parameters(),lr=LR)
	loss_func = nn.MSELoss()

	for epoch in range(100):
		h_state = None
		for step,(b_x,b_y) in enumerate(train_loader):
			b_x = b_y.view(1,TIME_STEP,-1).cuda()
			b_y = b_y.view(1,TIME_STEP,-1).cuda()

			prediction,h_state = gru(b_x,h_state)
			# h_state = h_state.data # ?

			loss = loss_func(prediction,b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (step+1)%30==0:
				print (epoch,step,loss.item(),end=" ")
				print (get_loss(valid_loader))

if __name__ == "__main__":
	train()

