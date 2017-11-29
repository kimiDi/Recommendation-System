import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from numpy.linalg import solve
%matplotlib inline

link = pd.read_csv('ml-latest-small/links.csv')
link.head()

movie = pd.read_csv('ml-latest-small/movies.csv')
movie.head()

rate = pd.read_csv('ml-latest-small/ratings.csv')
rate.head()

tag = pd.read_csv('ml-latest-small/tags.csv')
tag.head()

# rating matrix building
table = pd.pivot_table(rate, index = 'userId', columns = 'movieId', values = 'rating')
rating = table.fillna(value = 0).as_matrix()
rating

rating.shape()

# ALS algorithm
def myALS(lambda_, n_factors, ratings, n_iter):
	m, n = rating.shape
	X = 5 * np.random.rand(m, n_factors)
	Y = 5 * np.random.rand(n_factors, n)
	mse = []
	
	def get_mse(Q, X, Y):
		return np.square(Q - np.dot(X, Y)).mean()
	
	for ii in xrange(n_iterations):
		X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(n_factors), np.dot(Y, ratings.T)).T
		Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(n_factors), np.dot(X.T, ratings))
		if ii % 5 == 0:
			new_mse = get_mse(ratings, X, Y)
			mse.append(new_mse)
			print('{}th iteration'.format(ii))
			print('Train mse:' + str(new_mse))
	ratings_hat = np.dot(X, Y)
	plt.plot(mse)

lambda_ = 0.1
n_factors = 50
n_iterations = 31

myALS(lambda_, n_factors, rating, n_iterations)

# ALS Algorithm

class ExplicitMF():
	def __init__(self, ratings, n_factors = 40, item_reg = 0.0, user_reg = 0.0, verbose = False):
		self.ratings = ratings
		self.n_users, self.n_items = rating.shape
		self.n_factors = n_factors
		self.item_reg = item_reg
		self.user_reg = user_reg
		self._v = verbose

	def als_step(self, latent_vectors, fixed_vecs, ratings, _lambda, type = ‘user’):
		if type == 'user':
			YTY = fixed_vecs.T.dot(fixed_vecs)
			lambdaI = np.eye(YTY.shape[0]) * _lambda
			
			for u in xrange(latent_vectors.shape[0]):
				latent_vectors[u, :] = solve((YTY + lambdaI), rating[u, :].dot(fixed_vecs))
		elif type == 'item':
			XTX = fixed_vecs.T.dot(fixed_vecs)
			lambdaI = np.eye(XTX.shape[0]) * _lambda
			for i in xrange(latent_vectors.shape[0]):
				latent_vectors[i, :] = solve((XTX + lambdaI), ratings[:, i].T.dot(fixed_vecs))
		return latent_vectors
	
	def train(self, n_iter = 10):
		self.user_vecs = np.random.random((self.n_users, self.n_factors))
		self.item_vecs = np.random.random((self.n_items, self.n_factors))

		self.partial_train(n_iter)
	
	def partial_train(self, n_iter):
		ctr = 1
		while ctr <= n_iter:
			if ctr % 10 == 0 and self._v:
				print('\tcurrent iteration:{}'.format(ctr))
			self.user_vecs = self.als_step(self.user_vecs, self.item_vecs, self.ratings, self.user_reg, type = 'user')
			self.item_vecs = self.als_step(self.item_vecs, self.user_vecs, self.ratings, self.user_reg, type = 'item')
			ctr += 1
	
	def predict_all(self):
		predictions = np.zeros((self.user_vecs.shape[0], self.item_vecs.shape[0]))
		for u in range(self.user_vecs.shape[0]):
			for i in range(self.item_vecs.shape[0]):
				predictions[u, i] = self.predict(u, i)
		return predictions
	
	def predict(self, u, i):
		return self.user_vecs[u, :].dot(self.item_vecs[i,:].T)
	
	def get_mse(self, pred, y):
		return np.square(pred - y).mean()
	
	def calculate_learning_curve(self, iter_array):
		iter_array.sort()
		self.train_mse = []
		iter_diff = 0
		for (i, n_iter) in enumerate(iter_array):
			if self._v:
				print('Iterations:{}'.format(n_iter))
			if i == 0:
				self.train(n_iter - iter_diff)
			else:
				self.partial_train(n_iter - iter_diff)
			
			predictions = self.predict_all()
			
			self.train_mse += [self.get_mse(predictins, self.ratings)]
			
			if self._v:
				print('Train mse:' + str(self.train_mse[-1]))
			iter_diff = n_iter
		plt.plot(self.train_mse)
	
ExplicitMF(ratings = rating, n_factors= 50, item_reg = 0.2, user_reg = 0.2, verbose = True).calculate_learning_curve(np.arange(1, 31, 5))
