import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from math import sqrt
import sys 
from math import pi
# sys.path.append('../Erlangen/inverse-elliptical')
#from haarWavelet import *
# from mapOnInterval import *
np.random.seed(42)
# Domäne: x \in [0,1], t \in [0,1]

# Dynamik: Vektorfeld u(t, x)
# Zellen: x_i' = alpha*u(t, x_i(t)) + beta*W_i'(t) + grad(Phi(t, x_i(t)))

# konstanter Fluss nach rechts mit Geschwindigkeit 1
def u(x):
  return 5.0 + np.zeros_like(x)

N = 2000
h = 1/N

# virtuelles "Beobachtungsrauschen"
sigma = 0.01

# class Params:
# 	def __init__(self, alpha, beta, Phi_moi):
# 		self.alpha = alpha
# 		self.beta = beta
# 		self.Phi_moi = Phi_moi
# 		self.DPhi_moi = mapOnInterval("fourier", differentiateFourier(Phi_moi.fouriercoeffs))
# 	def Phi(self, x):  
# 		return self.Phi_moi.handle(x)
# 	def DPhi(self, x):
# 		return self.DPhi_moi.handle(x)

class Potential:
  def __init__(self, coeffs):
    self.coeffs = coeffs
    self.N = len(coeffs)
    assert self.N % 2 == 0
  
  def Phi(self, x):
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    # constant = self.coeffs[0]*np.ones((len(x),1))
    cosine_coeffs = (self.coeffs[0:(self.N)//2])[np.newaxis, :]
    modes = np.arange(1,(self.N)//2+1)
    cosines = cosine_coeffs*np.cos(2*pi*modes*x[:,np.newaxis])
    sine_coeffs = (self.coeffs[(self.N)//2:])[np.newaxis, :]
    sines = sine_coeffs*np.sin(2*pi*modes*x[:,np.newaxis])
    mode_matrix = np.hstack((cosines, sines))
    return np.sum(mode_matrix, axis=1)
  
  def DPhi(self, x):
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    constant = self.coeffs[0]*np.ones((len(x),1))
    cosine_coeffs = (self.coeffs[0:(self.N)//2])[np.newaxis, :]
    modes = np.arange(1,(self.N)//2+1)
    cosines = -2*pi*cosine_coeffs*modes*np.sin(2*pi*modes*x[:,np.newaxis])
    sine_coeffs = (self.coeffs[(self.N)//2:])[np.newaxis, :]
    sines = 2*pi*sine_coeffs*modes*np.cos(2*pi*modes*x[:,np.newaxis])
    mode_matrix = np.hstack((constant, cosines, sines))
    return np.sum(mode_matrix, axis=1)

true_coeffs = -0.1*np.array([0, -0.405, 0.072, -0.0085, 0.0012,0,0,0.1,0,0,0,0,0,0])
pot = Potential(true_coeffs)

xs_plot = np.linspace(0,1,500)
plt.figure()
plt.plot(xs_plot, pot.Phi(xs_plot))
# plt.plot(xs_plot, pot.DPhi(xs_plot))

#%%
    

# params_true = Params(0.5, 0.02, mapOnInterval("fourier", np.array([0, -0.015, 0, 0.015,0,0,0,0,0,0, 0.005, 0, 0,0,0,0,0,0,0.001])))

alpha = 0.05
beta = 0.2
# Zeitschritt ohne Brownsche Bewegung
def stepfnc_det(xk, pot):
	return xk + h*alpha*u(xk) - h*pot.DPhi(xk)

def stepfnc_noisy(xk, pot):
	return stepfnc_det(xk, pot) + sqrt(h)*beta*np.random.normal(0,1, xk.shape)
	
def trajectory(x0s, pot):
  xs = np.zeros((len(x0s), N))
  xs[:, 0] = x0s
  ts = np.linspace(0, 1, N)
  for k in range(1, N):
    t = ts[k]
    xs[:, k] = stepfnc_noisy(xs[:, k-1], pot)
  return ts, xs

def loglike_step(xkp1, xk, pot):
	return 1/(2*h*beta**2)*(xkp1 - stepfnc_det(xk, pot))**2


def loglike(xs, pot):
  l = 0
  for m in range(xs.shape[0]):
    for n in range(xs.shape[1]-1):
      l += loglike_step(xs[m, n+1], xs[m, n], pot)
  return l

def loglike_(xs, pot):
	l = 0
	for n in range(xs.shape[1]-1):
		l += np.sum(loglike_step(xs[:, n+1], xs[:, n], pot))
	return l

def loglike__(xs, pot):
  l = 0
  for m in range(xs.shape[0]):
    l += np.sum(loglike_step(xs[m, 1:], xs[m, 0:-1], pot))
  return l
  


xvals = np.linspace(0, 1, 100)

plt.figure(); 
plt.plot(xvals, pot.Phi(xvals))
#plt.plot(xvals, params_true.DPhi(xvals))
plt.title("Potential $\Phi$")
plt.show()

N_particles = 120
# Test des Vorwaertsmodelles
x0s = np.linspace(0, 0.6, N_particles)
ts, xs = trajectory(x0s, pot)

plt.figure(); 
for m, x0 in enumerate(x0s):
  plt.plot(ts, xs[m, :], label="Partikel " + str(m))
# plt.legend()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Trajectories of all particles")

# print(loglike_(xs, pot))
# print(loglike_(xs, Potential(np.random.normal(0,0.001,(14,)))))
# print(loglike_(xs, Potential(true_coeffs)))

# import time
# t1 = time.time()
# val1 = loglike(xs, pot)
# t2 = time.time()
# print(f"{val1} in {t2-t1}")
# val2 = loglike_(xs, pot)
# t3 = time.time()
# print(f"{val2} in {t3-t2}")
# val3 = loglike__(xs, pot)
# t4 = time.time()
# print(f"{val3} in {t4-t3}")


#%%
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
def optfnc(par, xs):
	return loglike__(xs, Potential(par))

import scipy.optimize as opt

res = opt.minimize(optfnc, np.zeros_like(true_coeffs), method="BFGS",  options={"disp": True})
#params_opt = Params(res.x[0], res.x[1], mapOnInterval("fourier", np.concatenate((np.array([0]), res.x[2:]))))
coeffs_opt = res.x

#plt.plot(res.x[0], res.x[1], '.m', markersize=10)

print("optimization: ")
print("coeffs = " + str(coeffs_opt))
print("actually = " + str(true_coeffs))

pot_opt = Potential(coeffs_opt)
plt.figure(); 
plt.plot(xvals, pot.Phi(xvals), label="true")
plt.plot(xvals, pot_opt.Phi(xvals), label="opt")
# plt.plot(xvals, params_true.DPhi(xvals))
plt.title("Potential $\Phi$")
plt.legend()
plt.show()

def MISE(pot):
  fncvals = pot.Phi(xs_plot)
  fncvals_true = Potential(true_coeffs).Phi(xs_plot)
  return np.trapz((fncvals-fncvals_true)**2, x=xs_plot)

# #%%

# # Test des Vorwaertsmodelles
# x0s = np.linspace(0, 0.6, 10)
# ts, xs_opt = trajectory(x0s, params_opt)

# plt.figure();
# plt.plot(xvals, params_opt.Phi(xvals))
# #plt.plot(xvals, params_true.DPhi(xvals))
# plt.title("Potential $\Phi$ (optimized)")



# plt.figure();
# for m, x0 in enumerate(x0s):
#   plt.plot(ts, xs_opt[m, :], label="Partikel " + str(m))
# # plt.legend()
# plt.xlabel("t")
# plt.ylabel("x")
# plt.title("Trajectories of all particles with optimal parameters")


#%%

num_particles_list = [2**n for n in range(3,7)]
num_MC = 3

MISE_array = np.zeros((len(num_particles_list), num_MC))

for n_p, num_p in enumerate(num_particles_list):
  print(f"num_particles = {num_p}")
  for m_MC in range(num_MC):
    print(f"MC iteration {m_MC}")
    # Test des Vorwaertsmodelles
    x0s = np.linspace(0, 0.6, num_p)
    ts, xs = trajectory(x0s, pot)
    res = opt.minimize(lambda par: optfnc(par, xs), np.zeros_like(true_coeffs), method="BFGS",  options={"disp": True})
    MISE_array[n_p,m_MC] = MISE(Potential(res.x))


plt.figure()
plt.boxplot(MISE_array.T, positions = num_particles_list)