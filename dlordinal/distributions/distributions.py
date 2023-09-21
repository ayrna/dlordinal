import math
import numpy as np
from scipy.special import gamma, softmax
from scipy.stats import binom, poisson
from scipy.special import hyp2f1

def beta_inc(a, b):
	"""Compute the incomplete beta function.
	
	Parameters
	----------
	a : float
		First parameter.
	b : float
		Second parameter.

	Returns
	-------
	beta_inc: float
		The value of the incomplete beta function.
	"""

	return (gamma(a) * gamma(b)) / gamma(a + b)

def beta(x, p, q, a = 1.0):
	"""Compute the beta density function for value x with parameters p, q and a.

	Parameters
	----------
	x : float
		Value to compute the density function for.
	p: float
		First parameter of the beta distribution.
	q: float
		Second parameter of the beta distribution.
	a: float, default=1.0
		Scaling parameter.

	Returns
	-------
	beta: float
		The value of the beta density function.
	"""

	return gamma(p+q) / (gamma(p) * gamma(q)) * a * x ** (a*p-1) * (1 - x**a) ** (q-1)

def beta_dist(x, p, q, a = 1.0):
	"""Compute the beta distribution function for value x with parameters p, q and a.

	Parameters
	----------
	x : float
		Value to compute the distribution function for.
	p: float
		First parameter of the beta distribution.
	q: float
		Second parameter of the beta distribution.
	a: float, default=1.0
		Scaling parameter.

	Returns
	-------
	beta: float
		The value of the beta distribution function.
	"""

	return (x ** (a*p)) / (p * beta_inc(p,q)) * hyp2f1(p, 1 - q, p + 1, x ** a)


def get_intervals(n):
	"""Get n evenly-spaced intervals in [0,1].

	Parameters
	----------
	n : int
		Number of intervals.

	Returns
	-------
	intervals: list
		List of intervals.
	"""

	points = np.linspace(1e-9, 1-1e-9, n + 1)
	intervals = []
	for i in range(0, points.size - 1):
		intervals.append((points[i], points[i+1]))

	return intervals

def get_beta_probabilities(n, p, q, a = 1.0):
	"""Get probabilities from beta distribution (p,q,a) for n splits.

	Parameters
	----------
	n : int
		Number of classes.
	p: float
		First parameter of the beta distribution.
	q: float
		Second parameter of the beta distribution.
	a: float, default=1.0
		Scaling parameter.

	Returns
	-------
	probs: list
		List of probabilities.
	"""

	intervals = get_intervals(n)
	probs = []

	# Compute probability for each interval (class) using the distribution function.
	for interval in intervals:
		probs.append(beta_dist(interval[1], p, q, a) - beta_dist(interval[0], p, q, a))

	return probs

def get_poisson_probabilities(n):
	"""Get probabilities from poisson distribution for n classes.

	Parameters
	----------
	n : int
		Number of classes.

	Returns
	-------
	probs: 2d array-like
		Matrix of probabilities where each row represents the true class
		and each column the probability for class n.
	"""

	probs = []

	for true_class in range(1, n+1):
		probs.append(poisson.pmf(np.arange(0, n), true_class))

	return softmax(np.array(probs), axis=1)

def get_binomial_probabilities(n):
	"""Get probabilities from binominal distribution for n classes.
	
	Parameters
	----------
	n : int
		Number of classes.

	Returns
	-------
	probs: 2d array-like
		Matrix of probabilities where each row represents the true class
		and each column the probability for class n.

	Example
	-------
	>>> get_binominal_probabilities(5)
	array([[6.561e-01, 2.916e-01, 4.860e-02, 3.600e-03, 1.000e-04],
		[2.401e-01, 4.116e-01, 2.646e-01, 7.560e-02, 8.100e-03],
		[6.250e-02, 2.500e-01, 3.750e-01, 2.500e-01, 6.250e-02],
		[8.100e-03, 7.560e-02, 2.646e-01, 4.116e-01, 2.401e-01],
		[1.000e-04, 3.600e-03, 4.860e-02, 2.916e-01, 6.561e-01]])
	"""

	params = {}
	
	params['4'] = np.linspace(0.1, 0.9, 4)
	params['5'] = np.linspace(0.1, 0.9, 5)
	params['6'] = np.linspace(0.1, 0.9, 6)
	params['7'] = np.linspace(0.1, 0.9, 7)
	params['8'] = np.linspace(0.1, 0.9, 8)
	params['10'] = np.linspace(0.1, 0.9, 10)
	params['12'] = np.linspace(0.1, 0.9, 12)
	params['14'] = np.linspace(0.1, 0.9, 14)

	probs = []

	for true_class in range(0, n):
		probs.append(binom.pmf(np.arange(0, n), n - 1, params[str(n)][true_class]))

	return np.array(probs)

def get_exponential_probabilities(n, p=1.0, tau=1.0):
	"""Get probabilities from exponential distribution for n classes.

	Parameters
	----------
	n : int
		Number of classes.
	p : float, default=1.0
		Exponent parameter.
	tau: float, default=1.0
		Scaling parameter.

	Returns
	-------
	probs: 2d array-like
		Matrix of probabilities where each row represents the true class
		and each column the probability for class n.
	
	Example
	-------
	>>> get_exponential_probabilities(5)
	array([[0.63640865, 0.23412166, 0.08612854, 0.03168492, 0.01165623],
       [0.19151597, 0.52059439, 0.19151597, 0.07045479, 0.02591887],
       [0.06745081, 0.1833503 , 0.49839779, 0.1833503 , 0.06745081],
       [0.02591887, 0.07045479, 0.19151597, 0.52059439, 0.19151597],
       [0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865]])
	"""

	probs = []

	for true_class in range(0, n):
		probs.append(-np.abs(np.arange(0, n) - true_class)**p / tau)

	return softmax(np.array(probs), axis=1)