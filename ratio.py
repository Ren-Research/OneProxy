import numpy as np
import copy
from numpy.linalg import inv
import pickle
import cvxpy as cp
import numpy as np
import pickle
from scipy import stats
import os
import random

from functions.solver import Solver

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error



#############################################################################
# read and process data

true_630 = pickle.load(open('./true_630.pickle', 'rb'))
true_640 = pickle.load(open('./true_640.pickle', 'rb'))
true_a76 = pickle.load(open('./true_a76.pickle', 'rb'))
true_vpu = pickle.load(open('./true_vpu.pickle', 'rb'))
# rice latency
rice_pixel3 = pickle.load(open('./rice_pixel3.pickle', 'rb'))
rice_edgegpu = pickle.load(open('./rice_edgegpu.pickle', 'rb'))
rice_edgetpu = pickle.load(open('./rice_edgetpu.pickle', 'rb'))
rice_eyeriss = pickle.load(open('./rice_eyeriss.pickle', 'rb'))
rice_fpga = pickle.load(open('./rice_fpga.pickle', 'rb'))
rice_raspi4 = pickle.load(open('./rice_raspi4.pickle', 'rb'))
# eagle latency
eagle_gtx = pickle.load(open('./eagle_gtx.pickle', 'rb'))
eagle_cpu_855 = pickle.load(open('./eagle_cpu_855.pickle', 'rb'))
eagle_dsp_855 = pickle.load(open('./eagle_dsp_855.pickle', 'rb'))
eagle_gpu_855 = pickle.load(open('./eagle_gpu_855.pickle', 'rb'))
eagle_i7 = pickle.load(open('./eagle_i7.pickle', 'rb'))
eagle_jetson = pickle.load(open('./eagle_jetson.pickle', 'rb'))
eagle_jetson_16 = pickle.load(open('./eagle_jetson_16.pickle', 'rb'))

# convert ms to s so that weight is smaller
true_vpu = [i/1000 for i in true_vpu]
true_630 = [i/1000 for i in true_630]
true_640 = [i/1000 for i in true_640]
true_a76 = [i/1000 for i in true_a76]
# eagle latency
eagle_gtx = [i/1000 for i in eagle_gtx]
eagle_cpu_855 = [i/1000 for i in eagle_cpu_855]
eagle_dsp_855 = [i/1000 for i in eagle_dsp_855]
eagle_gpu_855 = [i/1000 for i in eagle_gpu_855]
eagle_i7 = [i/1000 for i in eagle_i7]
eagle_jetson = [i/1000 for i in eagle_jetson]
eagle_jetson_16 = [i/1000 for i in eagle_jetson_16]
# rice latency
rice_pixel3 = [i/1000 for i in rice_pixel3]
rice_edgegpu = [i/1000 for i in rice_edgegpu]
rice_edgetpu = [i/1000 for i in rice_edgetpu]
rice_eyeriss = [i/1000 for i in rice_eyeriss]
rice_fpga = [i/1000 for i in rice_fpga]
rice_raspi4 = [i/1000 for i in rice_raspi4]

encode = pickle.load(open('./encode.pickle', 'rb'))
	
# proxy device, latency
proxy = true_vpu
target = eagle_gtx
# convert to numpy array
target = np.array(target).reshape(-1, 1)
proxy = np.array(proxy).reshape(-1, 1)

#print(target)
#print(proxy)

print("="*100)
print("SRCC between target and proxy: ", stats.spearmanr(proxy, target))
print("="*100)

encode = pickle.load(open('./encode.pickle', 'rb'))
encode = np.array(encode)
print(encode.shape)

#############################################################################
# directly use linear relationship
weight_linear = np.linalg.pinv(encode).dot(proxy)

print("="*100)
print("MSE for linear:", mean_squared_error(encode.dot(weight_linear), proxy))
print("SRCC between pred and true for linear:", stats.spearmanr(encode.dot(weight_linear), proxy))
#print(encode.dot(weight_linear))
#print(proxy)
#print(weight_linear)
print("="*100)

#############################################################################

#############################################################################
# transfer on target

# number of models used on target (out of n_train_proxy)
n_val_target = 10
n_train_target = 60

random.seed(1)
index = random.sample(range(encode.shape[0]), n_train_target+n_val_target)

x_train_target = np.zeros(shape=(n_train_target, encode.shape[1]))
y_train_target = np.zeros(shape=(n_train_target, 1))
for i in range(n_train_target):
	x_train_target[i, :] = encode[index[i]]
	y_train_target[i, :] = target[index[i]]

x_val_target = np.zeros(shape=(n_val_target, encode.shape[1]))
y_val_target = np.zeros(shape=(n_val_target, 1))
for i in range(n_val_target):
	x_val_target[i, :] = encode[index[i+n_train_target]]
	y_val_target[i, :] = target[index[i+n_train_target]]
	
	
#n_train_target_extra = 5
#
#random.seed(2)
#index = random.sample(range(encode.shape[0]), n_train_target_extra)
#x_train_target_extra = np.zeros(shape=(n_train_target_extra, encode.shape[1]))
#y_train_target_extra = np.zeros(shape=(n_train_target_extra, 1))
#for i in range(n_train_target_extra):
#	x_train_target_extra[i, :] = encode[index[i]]
#	y_train_target_extra[i, :] = target[index[i]]
#x_train_target = np.vstack((x_train_target, x_train_target_extra))
#y_train_target = np.vstack((y_train_target, y_train_target_extra))

	
x_all = np.vstack((x_train_target, x_val_target))
y_all = np.vstack((y_train_target, y_val_target))

x = copy.deepcopy(encode)
y = copy.deepcopy(target)

#print(x, x_train_target, x_val_target, x_all)

w = copy.deepcopy(weight_linear)
# bad
#w = np.linalg.pinv(x_train_target).dot(y_train_target)

solver = Solver(w, x_train_target, y_train_target, x_val_target, y_val_target, x, y, n_dim=31)

print("="*100)
print("SRCC between y_train: ", stats.spearmanr(x_train_target.dot(w), y_train_target))
print("SRCC between y_val: ", stats.spearmanr(x_val_target.dot(w), y_val_target))
print("SRCC between y_all: ", stats.spearmanr(x_all.dot(w), y_all))
print("="*100)

#############################################################################
# directly use linear relationship
weight_tmp = np.linalg.pinv(x_train_target).dot(y_train_target)

print("="*100)
print("MSE for linear:", mean_squared_error(x_train_target.dot(weight_tmp), y_train_target))
print("SRCC between pred and true for linear:", stats.spearmanr(encode.dot(weight_tmp), target))
#print(encode.dot(weight_linear))
#print(proxy)
print("="*100)

#############################################################################

#############################################################################
# transfer on target

norm = 2

lamb_dic = dict()
max_srcc_val = float('-inf')
max_srcc_all = None
max_srcc = None
max_lamb = None
MSE_val = []
MSE_all = []
MSE = []
SRCC_val = []
SRCC_all = []
SRCC = []
final_lat_val = None
final_lat_all = None
final_lat = None
#lamb_range = np.arange(0, 10.01, 0.01)
lamb_range = np.arange(1, 100000, 0.001)

for lamb in lamb_range:
	estimate_target_train, estimate_target_val, estimate_target_all, estimate_target, srcc_train, srcc_val, srcc_all, srcc = solver.solve(lamb, norm)

	if srcc_val > max_srcc_val:
		max_srcc_val = srcc_val
		max_srcc_all = srcc_all
		max_srcc = srcc
		max_lamb = lamb
		final_lat_val = estimate_target_val
		final_lat_all = estimate_target_all
		final_lat = estimate_target
	print("lamb:", lamb, "srcc_val:", srcc_val, "srcc_all:", srcc_all, "srcc_train:",  srcc_train, "srcc", srcc)
	
	mse_val = mean_squared_error(y_val_target, estimate_target_val)
	MSE_val.append(mse_val)
	mse_all = mean_squared_error(y_all, estimate_target_all)
	MSE_val.append(mse_all)
	mse = mean_squared_error(target, estimate_target)
	MSE.append(mse)
	
	SRCC_val.append(srcc_val)
	SRCC_all.append(srcc_all)
	SRCC.append(srcc)
	
	print("Max SRCC val:", max_srcc_val, "lambda:", max_lamb, "MSE train:",  mean_squared_error(y_train_target, estimate_target_train), "SRCC train:", srcc_train, "MSE val:", mse_val, "SRCC all:", max_srcc_all, "MSE all:", mse_all, "SRCC 15625:", max_srcc, "MSE 15625:", mse)
	print()

res = dict()
res['max_lamb'] = max_lamb
res['max_srcc_val'] = max_srcc_val
res['max_srcc_all'] = max_srcc_all
res['max_srcc'] = max_srcc
res['mse_val'] = MSE_val
res['mse_all'] = MSE_all
res['mse'] = MSE
res['lambda'] = lamb_range
res['srcc_val'] = SRCC_val
res['srcc_all'] = SRCC_all


# train with both training and validation data
x_all = np.vstack((x_train, x_val))
y_all = np.vstack((y_train, y_val))
solver = Solver(w, x_all, y_all, x_all, y_all, x, y, n_dim=x_all.shape[1])	# x and jetson are 15625 data
estimate_target_all, _, estimate_target, srcc_all, _, srcc = solver.solve(max_lamb, norm)
print("Final SRCC on both train and val:", srcc_all, "Final SRCC on 15625:", srcc)

#res['final_lat'] = estimate_target
res['final_srcc_all'] = srcc_all
res['final_srcc'] = srcc


dic = dict()
for i in range(len(model_pool)):
	#print(model_pool[i])
	dic[str(model_pool[i])] = estimate_target[i, :][0]
	
res['final_lat'] = dic	# store as arch: lat
res['true_lat'] = y
res['weight'] = solver.get_weight()
	

with open('./data/' + target + '_' + str(n_train) + '.pickle', 'wb') as handle:
	pickle.dump(res, handle)
	
print(len(model_pool))
if not os.path.exists('./data/' + proxy + '.pickle'):
	dic = dict()
	for i in range(len(model_pool)):
		#print(model_pool[i])
		dic[str(model_pool[i])] = lat_proxy[i]
		
	with open('./data/' + proxy + '.pickle', 'wb') as handle:
		pickle.dump(dic, handle)
		
dic = dict()
for i in range(len(x)):
	#print(model_pool[i])
	dic[str(model_pool[i])] = lat_proxy[i]
	
with open('./data/' + proxy + '.pickle', 'wb') as handle:
	pickle.dump(dic, handle)
	
	
lat_target = []

for sample in model_pool:
	lat_target.append(target_predcitor.predict_efficiency(sample)[0])
	
with open('./data/' + target + '.pickle', 'wb') as handle:
	pickle.dump(lat_target, handle)