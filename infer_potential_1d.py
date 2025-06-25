import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from math import sqrt
import sys 
sys.path.append('../Erlangen/inverse-elliptical')
#from haarWavelet import *
from mapOnInterval import *

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
	def __init__(self, alpha, beta, Phi_moi):
		self.alpha = alpha
		self.beta = beta
		self.Phi_moi = Phi_moi
		self.DPhi_moi = mapOnInterval("fourier", differentiateFourier(Phi_moi.fouriermodes))
	def Phi(self, x):  
		return self.Phi_moi.handle(x)
	def DPhi(self, x):
		return self.DPhi_moi.handle(x)


params_true = Params(0.5, 0.02, mapOnInterval("fourier", np.array([0, -0.015, 0, 0.015,0,0,0,0,0,0, 0.005, 0, 0,0,0,0,0,0,0.001])))

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

def loglike_(xs, params):
	l = 0
	for n in range(xs.shape[1]-1):
		l += np.sum(loglike_step(xs[:, n+1], xs[:, n], params))
	return l



xvals = np.linspace(0, 1, 100)

plt.figure(); plt.ion()
plt.plot(xvals, params_true.Phi(xvals))
#plt.plot(xvals, params_true.DPhi(xvals))
plt.title("Potential $\Phi$")
plt.show()


# Test des Vorwaertsmodelles
x0s = np.linspace(0, 0.6, 10)
ts, xs = trajectory(x0s, params_true)

plt.figure(); 
for m, x0 in enumerate(x0s):
  plt.plot(ts, xs[m, :], label="Partikel " + str(m))
plt.legend()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Trajectories of all particles")
"""
# try inference
N1 = 20
N2 = 20
alphas = np.linspace(-0.1, 0.1, N1)
betas = np.linspace(-0.1, 0.1, N2)



# try inference
cst3 = np.zeros((N1,N2))
for m, alpha in enumerate(alphas):
	for n, beta in enumerate(betas):
		params = Params(0.5, 0.02, mapOnInterval("fourier", np.array([0, alpha, beta, 0, 0, 0, 0])))
		cst3[m, n] = loglike(xs, params)

ind = np.unravel_index(np.argmin(cst3, axis=None), cst3.shape)
alpha_gridsearch = alphas[ind[0]]
beta_gridsearch  = betas[ind[1]]
params_gridsearch =  Params(0.5, 0.02, mapOnInterval("fourier", np.array([0, alpha_gridsearch, beta_gridsearch, 0, 0, 0, 0])))

print("grid search: ")
print("alpha = " + str(alpha_gridsearch))
print("beta = " + str(beta_gridsearch))

plt.figure();
plt.contourf(alphas, betas, cst3.T, 50)
plt.plot(-0.015, 0.015, '.w', markersize=10)
plt.plot(alpha_gridsearch, beta_gridsearch, '.y', markersize=10)"""


# optimization procedure
def optfnc(par):
	params = Params(par[0], par[1], mapOnInterval("fourier", np.concatenate((np.array([0]), par[2:]))))
	return loglike_(xs, params)

import scipy.optimize as opt

res = opt.minimize(optfnc, np.array([0.1, 0.1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0]), method="L-BFGS-B", bounds=((None, None), (0.00001, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None)),  options={"disp": True})
params_opt = Params(res.x[0], res.x[1], mapOnInterval("fourier", np.concatenate((np.array([0]), res.x[2:]))))

#plt.plot(res.x[0], res.x[1], '.m', markersize=10)

print("optimization: ")
print("alpha = " + str(res.x[0]))
print("beta = " + str(res.x[1]))

# Test des Vorwaertsmodelles
x0s = np.linspace(0, 0.6, 10)
ts, xs_opt = trajectory(x0s, params_opt)

plt.figure();
plt.plot(xvals, params_opt.Phi(xvals))
#plt.plot(xvals, params_true.DPhi(xvals))
plt.title("Potential $\Phi$ (optimized)")



plt.figure();
for m, x0 in enumerate(x0s):
  plt.plot(ts, xs_opt[m, :], label="Partikel " + str(m))
plt.legend()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Trajectories of all particles with optimal parameters")
