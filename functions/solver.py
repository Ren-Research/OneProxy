import pickle
import numpy as np

import cvxpy as cp

from scipy import stats

#res = pickle.load(open('./weight_tune_res.pickle','rb'))
#
#w = res['weight_tune']

class Solver:
	def __init__(self, w, train_x, train_y, val_x, val_y, x, y, n_dim=600):
		self.n_dim = n_dim
		self.w = w
		self.train_x = train_x
		self.train_y = train_y
		self.val_x = val_x
		self.val_y = val_y
		self.x = x
		self.y = y
		
	def solve(self, lamb, norm):
		Ap = self.w
		#print("w", Ap)

		b = cp.Variable((self.n_dim, 1))
		#a = cp.Variable((100, 1))
		a = cp.Variable()
		A = cp.vstack([a for _ in range(self.n_dim)])
		c = cp.Variable()
		C = cp.vstack([c for _ in range(len(self.train_x))])
		I = cp.vstack([1 for _ in range(self.n_dim)])
		
		if norm != 'inf':
		#objective = cp.Minimize(cp.sum_squares(x @ (cp.multiply(Ap, cp.multiply(A, I) + b)) + C - y) / len(x) + lamb*cp.norm(b, 2))
			objective = cp.Minimize(cp.sum_squares(self.train_x @ (cp.multiply(Ap, A + b)) + C - self.train_y) / len(self.train_x) + lamb*cp.norm(b, norm))
		else:
			objective = cp.Minimize(cp.sum_squares(self.train_x @ (cp.multiply(Ap, A + b)) + C - self.train_y) / len(self.train_x) + lamb*cp.pnorm(b, 'inf'))
		
		prob = cp.Problem(objective)
		result = prob.solve()#verbose=True)
		
		C = np.array([c.value for _ in range(len(self.train_x))]).reshape(-1, 1)
		A = np.array([a.value for _ in range(self.n_dim)]).reshape(-1, 1)
		#print(C, A)
		estimate_target_train = self.train_x.dot(np.multiply(Ap, A + b.value)) + C
		srcc_train = stats.spearmanr(self.train_y, estimate_target_train)[0]
		
		C = np.array([c.value for _ in range(len(self.val_x))]).reshape(-1, 1)
		A = np.array([a.value for _ in range(self.n_dim)]).reshape(-1, 1)
		print(lamb)
#		print("w", Ap)
#		print("A", A)
#		print("c", C)
#		print("b", b.value)
#		print("b norm", np.linalg.norm(b.value, ord=1))

		#estimate_target = arch_encode.dot(np.multiply(Ap, np.multiply(A, I.value) + b.value)) + C
		estimate_target_val = self.val_x.dot(np.multiply(Ap, A + b.value)) + C
		
		srcc_val = stats.spearmanr(self.val_y, estimate_target_val)[0]
		
		all_x = np.vstack((self.train_x, self.val_x))
		C = np.array([c.value for _ in range(len(all_x))]).reshape(-1, 1)
		A = np.array([a.value for _ in range(self.n_dim)]).reshape(-1, 1)
		estimate_target_all = all_x.dot(np.multiply(Ap, A + b.value)) + C
		all_y = np.vstack((self.train_y, self.val_y))
		srcc_all = stats.spearmanr(all_y, estimate_target_all)[0]
		
		C = np.array([c.value for _ in range(len(self.x))]).reshape(-1, 1)
		A = np.array([a.value for _ in range(self.n_dim)]).reshape(-1, 1)
		estimate_target = self.x.dot(np.multiply(Ap, A + b.value)) + C
		srcc = stats.spearmanr(self.y, estimate_target)[0]
		
		self.weight = np.multiply(Ap, A + b.value)
		
		#print(self.weight)
		
		return estimate_target_train, estimate_target_val, estimate_target_all, estimate_target, srcc_train, srcc_val, srcc_all, srcc
		
	def get_weight(self):
		return self.weight