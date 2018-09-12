"""
定义了一些常用的类和函数

如 损失 等
"""
import torch
import torch.nn as nn


class MSELoss_with_mask(nn.Module):
	"""
	全部数据上与缺失部分上MSE的加权平均
	"""
	def __init__(self,weight=(1,1)):
		"""
		weight 全部数据与缺失部分的权重
		"""
		super(MSELoss_with_mask,self).__init__()
		self.weight = weight
	
	def forward(self,y_predict,y_target,mask):
		
		one_num  = torch.sum(mask)
		zero_num = torch.sum(1-mask)
		mse  = nn.MSELoss()
		loss1 = mse(y_predict, y_target)
		loss2 = mse(y_predict*(1-mask),
					y_target *(1-mask)) * (zero_num+one_num)/zero_num
		w1,w2 = self.weight
		loss = (loss1*w1+loss2*w2)/(w1+w2)
		return loss


class MAE(nn.Module):
	"""
	MAE: Mean Absolute Error,平均绝对损失
	仅考虑缺失部分的MAE 
	"""
	def __init__(self):
		super(MAE,self).__init__()

	def forward(self,y_predict,y_target,mask):
		one_num  = torch.sum(mask)
		zero_num = torch.sum(1-mask)
		L1   = nn.L1Loss()
		loss = L1(y_predict*(1-mask),
				  y_target *(1-mask)) *zero_num/(zero_num+one_num)
		return loss
