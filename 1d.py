import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from math import sqrt

# Domäne: x \in [0,1], t \in [0,1]

# Dynamik: Vektorfeld u(t, x)
# Zellen: x_i' = alpha*u(t, x_i(t)) + beta*W_i'(t) + grad(Phi(t, x_i(t)))

# konstanter Fluss nach rechts mit Geschwindigkeit 1
def u(x):
  return 1.0 + np.zeros_like(x)

N = 800
h = 1/N

# virtuelles "Beobachtungsrauschen"
sigma = 0.01

class Params:
	def __init__(self, alpha, beta, a, pos_a, s_a, b, pos_b, s_b):
		self.alpha = alpha
		self.beta = beta
		self.a = a
		self.pos_a = pos_a
		self.s_a = s_a
		self.b = b
		self.pos_b = pos_b
		self.s_b = s_b
	def Phi(self, x):  
		return self.a*np.exp(-(x-self.pos_a)**2/(2*self.s_a)) - self.b*np.exp(-(x-self.pos_b)**2/(2*self.s_b))
	def DPhi(self, x):
		return self.a*np.exp(-(x-self.pos_a)**2/(2*self.s_a))*(self.pos_a-x)/self.s_a - self.b*np.exp(-(x-self.pos_b)**2/(2*self.s_b))*(self.pos_b-x)/self.s_b


#params_true = Params(0.5, 0.02, 0.01, 0.25, 0.001, 0.02, 0.7, 0.004)
params_true = Params(0.5, 0.02, 0.02, 0.25, 0.001, 0.04, 0.7, 0.004)

# Zeitschritt ohne Brownsche Bewegung
def stepfnc_det(xk, params):
	return xk + h*params.alpha*u(xk) - h*params.DPhi(xk)

def stepfnc_noisy(xk, params):
	return stepfnc_det(xk, params) + sqrt(h)*params.beta*np.random.normal(0,1, xk.shape)
	
def trajectory(x0s, params):
  xs = np.zeros((len(x0s), N))
  xs[:, 0] = x0s
  ts = np.linspace(0, 1, N)
  for k in range(1, N):
    t = ts[k]
    xs[:, k] = stepfnc_noisy(xs[:, k-1], params)
  return ts, xs

def loglike_step(xkp1, xk, params):
	return 1/(2*h*params.beta**2)*(xkp1 - stepfnc_det(xk, params))**2 + np.log(sqrt(h)*params.beta)


def loglike(xs, params):
	for m in range(xs.shape[0]):
		l = 0
		for n in range(xs.shape[1]-1):
			l += loglike_step(xs[m, n+1], xs[m, n], params)
	return l

#def loglike_step(x, d, params):
#	return 1/(2*sigma**2)*(x-d)**2 ##
#
#def loglike(xs, data, params):
#	for m in range(xs.shape[0]):
#		l = 0
#		for n in range(len(xs)-1):
#			l += loglike_step(xs[m, n], data[m, n], params)
#	return l
		


xvals = np.linspace(0, 1, 100)

plt.figure(); plt.ion()
plt.plot(xvals, params_true.Phi(xvals))
#plt.plot(xvals, GradPhi(xvals))
plt.title("Potential $\Phi$")
plt.show()


# Test des Vorwaertsmodelles
x0s = np.linspace(0, 0.6, 3)
ts, xs = trajectory(x0s, params_true)

plt.figure();
for m, x0 in enumerate(x0s):
  plt.plot(ts, xs[m, :], label="Partikel " + str(m))
plt.legend()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Trajectories of all particles")


# try inference
# N1 = 30
# N2 = 30
# alphas = np.linspace(0.0, 1.0, N1)
# betas = np.linspace(0.0, 1.0, N2)



# # try inference
# cst3 = np.zeros((N1,N2))
# for m, alpha in enumerate(alphas):
# 	for n, beta in enumerate(betas):
# 		params = Params(0.5, 0.02, 0.01, alpha, 0.001, 0.02, beta, 0.004)
# 		cst3[m, n] = loglike(xs, params)

# ind = np.unravel_index(np.argmin(cst3, axis=None), cst3.shape)
# alpha_gridsearch = alphas[ind[0]]
# beta_gridsearch  = betas[ind[1]]
# params_gridsearch =  Params(0.5, 0.02, 0.01, alpha_gridsearch, 0.001, 0.02, beta_gridsearch, 0.004)

# print("grid search: ")
# print("alpha = " + str(alpha_gridsearch))
# print("beta = " + str(beta_gridsearch))

# plt.figure();
# plt.contourf(alphas, betas, cst3.T, 50)
# plt.plot(params_true.alpha, params_true.beta, '.w', markersize=10, label="true")
# plt.plot(alpha_gridsearch, beta_gridsearch, '.y', markersize=10, label="grid search")


# optimization procedure
def optfnc(par):
	params = Params(0.5, 0.02, 0.02, par[0], 0.001, 0.04, par[1], 0.004)
	return loglike(xs, params)

import scipy.optimize as opt

res = opt.minimize(optfnc, np.array([1.0,1.0]), method="L-BFGS-B", bounds=((None, None), (None, None)), options={"disp": False})
params_opt = Params(0.5, 0.02, 0.01, res.x[0], 0.001, 0.02, res.x[1], 0.004)


plt.plot(res.x[0], res.x[1], '.m', markersize=10, label="opt")
plt.legend()
print("optimization: ")
print("alpha = " + str(res.x[0]))
print("beta = " + str(res.x[1]))

# Test des Vorwaertsmodelles
x0s = np.linspace(0, 0.6, 10)
ts, xs_opt = trajectory(x0s, params_opt)

plt.figure();
for m, x0 in enumerate(x0s):
  plt.plot(ts, xs_opt[m, :], label="Partikel " + str(m))
plt.legend()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Trajectories of all particles with optimal parameters")
plt.show()

input()
