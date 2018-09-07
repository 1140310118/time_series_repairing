import copy
import torch
import torch.utils.data as Data
import numpy as np
import threading


def make_data(data,time_step,batch_size,dimension,missing_rate=0.2):
	"""
	将时间序列进行缺失处理，返回
	  - X 缺失的时间序列及缺失的位置
	  - Y 完整的时间序列
	"""
	Y = data.reshape(batch_size,time_step,dimension)
	Y_copy = copy.deepcopy(Y)

	X = []
	masks = []
	for i in range(time_step):
		if not X: # 首次循环，不缺失
			x = Y_copy[:,0]
			mask = np.ones_like(x)
		else:
			x = Y_copy[:,i]
			mask = (np.random.random(x.shape) > missing_rate).astype(np.int32)
			x = x*mask + X[-1]*(1-mask)
		masks.append(mask)	
		X.append(x)

	masks = np.array(masks)
	masks = np.swapaxes(masks,0,1) # 交换前两个维度，使得矩阵的形状变为 (batch_size,time_step,dimension) 
	X = np.array(X)
	X = np.swapaxes(X,0,1) # 同上
	# X = np.concatenate((X,masks),axis=2)
	
	return X,Y,masks

def make_dataloader(X,Y,masks,batch_size):
	X,Y,masks = torch.FloatTensor(X),torch.FloatTensor(Y),torch.FloatTensor(masks)
	torch_dataset = Data.TensorDataset(X,Y,masks)
	dataloader = Data.DataLoader(
		dataset=torch_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=2,
	)
	return dataloader

def get_batch(np_data,time_step,batch_size,dimension,missing_rate):
	"""
	返回一个dataloader
	for b_x,b_y in dataloader:
		b_x (batch_size,time_step,dimension*2)
		b_y (batch_size,time_step,dimension)
	"""
	num = np_data.shape[0]
	X,Y,masks = make_data(np_data,time_step,int(num/time_step),dimension,missing_rate)
	dataloader = make_dataloader(X,Y,masks,batch_size)
	return dataloader


if __name__ == "__main__":
	data_train = np.load('ts_data/data_train.npy')
	data_valid = np.load('ts_data/data_valid.npy')
	
	TIME_STEP  = 10
	BATCH_SIZE = 100
	dimension  = 66
	missing_rate = 0.2
	
	train_dataloader = get_batch(data_train,TIME_STEP,BATCH_SIZE,dimension,missing_rate)
	valid_dataloader = get_batch(data_valid,TIME_STEP,BATCH_SIZE,dimension,missing_rate)
	
	

