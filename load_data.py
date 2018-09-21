"""
务必保证内存大于8G
"""
import os
import copy
import torch
import torch.utils.data as Data
import numpy as np



def makedir_if_not_exist(dir_):
	if not os.path.isdir(dir_):
		os.makedirs(dir_)


def _missing(x,x_last,delta_last,missing_rate=0.2):
	"""
	x,x_last,delta的形状为 (-1,dimension)
	随机缺失，供missing调用
	"""
	mask = (np.random.random(x.shape)>missing_rate).astype(np.uint8)
	x_missing = x*mask + x_last*(1-mask) # 缺失时，使用上一个时刻的值
	# 不缺失时(m=1)，d的值为1；缺失时(m=0)，d值为上次的值+1
	delta = mask + (1-mask)*(np.ones_like(mask)+delta_last)
	return x_missing,mask,delta

def missing(X,time_step):
	"""
	X 的形状为 (-1,time_step,dimension)
	"""
	X_missing = []
	Mask  = []
	Delta = []

	for i in range(time_step):
		x = X[:,0] # (-1,dimension)
		if not i:  # i=0
			x_missing = x
			x_shape   = x.shape
			mask  = np.ones(x_shape)
			delta = np.zeros(x_shape)
		else:
			x_missing,mask,delta = _missing(x,X_missing[-1],Delta[-1])
		X_missing.append(x_missing) 
		Mask.append(mask)
		Delta.append(delta)

	X_missing = np.array(X_missing) # (time_step,-1,dimension)
	Mask  = np.array(Mask,dtype=np.uint8)
	Delta = np.array(Delta,dtype=np.uint8)

	X_missing = np.swapaxes(X_missing,0,1) # (-1,time_step,dimension)
	Mask = np.swapaxes(Mask,0,1)
	Delta = np.swapaxes(Delta,0,1)

	return X_missing,Mask,Delta


def make_data(data,time_step,dimension,
		missing_rate=0.2,has_delta=False):
	"""
	将时间序列进行缺失处理，返回
	  - Xs 缺失的时间序列 (-1,time_step,dimension)
	  - Ys 下一个时刻的时间序列 (-1,1,dimension)
	  - Masks  缺失的位置 (-1,time_step,dimension)
	  - Deltas 延时，即距上一个的未缺失值的时间间隔，
	  	       用于在GRU-D中计算衰减系数
	  	       (-1,time_step,dimension)
	"""
	len_ = data.shape[0]
	Xs = []
	Ys = []
	Masks  = []
	Deltas = []

	for i in range(time_step): # 模拟窗口滑动
		num = (len_-i-1) // time_step # 切片个数  _i_######1
		_len = num*time_step # 长度
		# (-1,time_step,dimension)
		X = copy.deepcopy(data[i:_len+i]).reshape(-1,time_step,dimension) 
		# Y = copy.deepcopy(data[i+1:_len+i+1]).reshape(-1,time_step,dimension)[:,-2:-1]
		Y = copy.deepcopy(X)

		X_missing,Mask,Delta = missing(X,time_step)

		Xs.append(X_missing)
		Ys.append(Y)
		Masks.append(Mask)
		Deltas.append(Delta)

	Xs = np.array(Xs).reshape(-1,time_step,dimension)
	# Ys = np.array(Ys).reshape(-1,1,dimension)
	Ys = np.array(Ys).reshape(-1,time_step,dimension)
	Masks  = np.array(Masks, dtype=np.uint8).reshape(-1,time_step,dimension)
	Deltas = np.array(Deltas,dtype=np.uint8).reshape(-1,time_step,dimension)

	return Xs,Ys,Masks,Deltas


def make_dataloader(X,Y,Mask,Delta,batch_size,shuffle=True):
	"""
	需要画图的时候，可以这样调用此函数
		make_dataloader(X,Y,Mask,Delta,batch_size=1,shuffle=False)
	"""
	X,Y,Mask,Delta = torch.FloatTensor(X),torch.FloatTensor(Y),torch.FloatTensor(Mask),torch.FloatTensor(Delta)

	torch_dataset = Data.TensorDataset(X,Y,Mask,Delta)
	
	dataloader = Data.DataLoader(
		dataset=torch_dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=2,
	)
	return dataloader


def get_batch(source_dir,num,time_step,batch_size,dimension,
		missing_rate,shuffle=True,tmp_dir='data/tmp/',saved=True):
	"""
	返回一个dataloader
	for b_x,b_y,mask,delta in dataloader:
		b_x,b_y,mask,delta (batch_size,time_step,dimension)
	
	num     为仅使用前num条数据
	tmp_dir 为中间结果的保存地址
	saved   为真，则直接从指定地址中加载数据；为假，则重新生成数据，并将生成的数据保存 
	"""
	makedir_if_not_exist(tmp_dir)
	if not saved:
		np_data = np.load(source_dir)[:num]
		X,Y,Mask,Delta = make_data(np_data,time_step,
			dimension,missing_rate)
		dataloader = make_dataloader(X,Y,Mask,Delta,batch_size,shuffle)
		torch.save(dataloader,tmp_dir+'dataloader')
	else:
		dataloader = torch.load(tmp_dir+'dataloader')
	
	return dataloader


if __name__ == "__main__":
	data_train_dir = 'ts_data/data_train.npy'
	data_valid_dir = 'ts_data/data_valid.npy'
	
	TIME_STEP  = 10
	BATCH_SIZE = 100
	DIMSENSION  = 66
	missing_rate = 0.8
	
	train_dataloader = get_batch(data_train_dir,10000,
		TIME_STEP,BATCH_SIZE,DIMSENSION,missing_rate,
		tmp_dir="data/tmp/train/",saved=False)
	
	valid_dataloader = get_batch(data_valid_dir,5000,
		TIME_STEP,BATCH_SIZE,DIMSENSION,missing_rate,
		tmp_dir='data/tmp/valid/',saved=False)
	
	# save=True  大约需要 5~6s
	# save=False 大约需要 63 s
	

