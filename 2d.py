
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from math import sqrt

# Domäne: x \in [0,1]x[0,1], t \in [0,1]

# Dynamik: Vektorfeld u(t, x)
# Zellen: x_i' = alpha*u(t, x_i(t)) + beta*W_i'(t) + grad(Phi(t, x_i(t)))

N_pts = 31
def u(x):
  return (np.array([[0,1],[-1,0]])  @ x) + np.array([0.5*x[0]*np.cos(15*x[1]**2), 0.2*x[0]**2])

alpha = 0.5
beta = 0.05

N = 800
T = 10
h = T/N
xs_vectors = np.linspace(-2, 2, N_pts)
ys_vectors = np.linspace(-2, 2, N_pts)
us = np.zeros((N_pts,N_pts))
vs = np.zeros((N_pts,N_pts))
for m, x in enumerate(xs_vectors):
	for n, y in enumerate(ys_vectors):
		pt = np.array([xs_vectors[m], ys_vectors[n]])
		us[m, n] = u(pt)[0]
		vs[m, n] = u(pt)[1]
		
XS_vectors, YS_vectors = np.meshgrid(xs_vectors, ys_vectors, indexing='ij')

eps = 1
tau1 = 0.05

def Phi1(x):
	return -0.1/((x[0]**2+x[1]**2)/(2*tau1**2)+eps)

def Phi2(x):
	return 1/(((x[0]-0.8)**2+(x[1]+0.8)**2)/(2*tau1**2)+eps)

def Phi(x):
	return Phi1(x)+Phi2(x)

def DPhi1(x):
	return 0.1/((x[0]**2+x[1]**2)/(2*tau1**2)+eps)**2 *np.array([x[0],x[1]])/tau1**2

def DPhi2(x):
	return -1/(((x[0]-0.8)**2+(x[1]+0.8)**2)/(2*tau1**2)+eps)**2 *np.array([x[0]-0.8,x[1]+0.8])/tau1**2

def DPhi(x):
	return DPhi1(x) + DPhi2(x)

N_pts_contour = 101
xs_contour = np.linspace(-2, 2, N_pts_contour)
ys_contour = np.linspace(-2, 2, N_pts_contour)
XS_contour, YS_contour = np.meshgrid(xs_contour, ys_contour, indexing='ij')
DPhivals1 = np.zeros((N_pts_contour,N_pts_contour))
DPhivals2 = np.zeros((N_pts_contour,N_pts_contour))
Phis = np.zeros((N_pts_contour,N_pts_contour))
for m, x in enumerate(xs_contour):
	for n, y in enumerate(ys_contour):
		pt = np.array([xs_contour[m], ys_contour[n]])
		DPhivals1[m,n] = DPhi(pt)[0]
		DPhivals2[m,n] = DPhi(pt)[1]
		Phis[m,n] = Phi(pt)

#plt.figure(2);plt.quiver(XS_contour, YS_contour, DPhivals1, DPhivals2); 

plt.figure(1);plt.ion();
plt.contourf(XS_contour, YS_contour, Phis, cmap=plt.get_cmap("YlGn"))
plt.quiver(XS_vectors, YS_vectors, us, vs); plt.show()
def stepfnc_det(xk):
	return xk + h*alpha*u(xk) - h*DPhi(xk)

def stepfnc_noisy(xk):
	return stepfnc_det(xk) + sqrt(h)*beta*np.random.normal(0,1, xk.shape)
	
def trajectory(x0):
  xs = np.zeros((len(x0), N))
  xs[:, 0] = x0
  ts = np.linspace(0, T, N)
  for k in range(1, N):
    t = ts[k]
    xs[:, k] = stepfnc_noisy(xs[:, k-1])
  return ts, xs

ts, xs = trajectory(np.array([0.5, 0.5]))

plt.figure(1)
plt.plot(xs[0,:], xs[1,:])
plt.plot(xs[0,0], xs[1,0], '.k', markersize=10)
ts, xs = trajectory(np.array([0.2, 0.5]))
plt.plot(xs[0,:], xs[1,:])
plt.plot(xs[0,0], xs[1,0], '.k', markersize=10)
ts, xs = trajectory(np.array([0.7, 0.5]))
plt.plot(xs[0,:], xs[1,:])
plt.plot(xs[0,0], xs[1,0], '.k', markersize=10)
ts, xs = trajectory(np.array([0.8, 0.5]))
plt.plot(xs[0,:], xs[1,:])
plt.plot(xs[0,0], xs[1,0], '.k', markersize=10)
ts, xs = trajectory(np.array([0.75, 0.5]))
plt.plot(xs[0,:], xs[1,:])
plt.plot(xs[0,0], xs[1,0], '.k', markersize=10)
ts, xs = trajectory(np.array([0.3, 0.5]))
plt.plot(xs[0,:], xs[1,:])
plt.plot(xs[0,0], xs[1,0], '.k', markersize=10)
ts, xs = trajectory(np.array([0.9, 0.5]))
plt.plot(xs[0,:], xs[1,:])	
plt.plot(xs[0,0], xs[1,0], '.k', markersize=10)
plt.show()
plt.pause(5)

#############################################

# virtuelles "Beobachtungsrauschen"
sigma = 0.01

class Params:
	def __init__(self, alpha, beta, a, s_a):
		self.alpha = alpha
		self.beta = beta
		self.a = a
		self.s_a = s_a
	def Phi(self, x):  
		return self.a*np.exp(-((x[0]-1)**2+(x[1]-1)**2)/(2*self.s_a))
	def DPhi(self, x):
		return self.Phi(x)*(-(x-1)/self.s_a)


params_true = Params(0.5, 0.02, 0.01, 0.3)

x0s = np.stack([np.linspace(0, 1, 10), np.zeros((10,))])

# Zeitschritt ohne Brownsche Bewegung
def stepfnc_det(xk, params):
	return xk + h*params.alpha*u(xk) - h*params.DPhi(xk)

def stepfnc_noisy(xk, params):
	return stepfnc_det(xk, params) + sqrt(h)*params.beta*np.random.normal(0,1, xk.shape)

def trajectory_parms(x0, params):
  xs = np.zeros((len(x0), N))
  xs[:, 0] = x0
  ts = np.linspace(0, 1, N)
  for k in range(1, N):
    t = ts[k]
    xs[:, k] = stepfnc_noisy(xs[:, k-1], params)
  return ts, xs

x0s = np.linspace(0, 0.6, 10)
y0s = np.zeros((10,))
z0s = np.stack([x0s, y0s])
zs = np.zeros((N, z0s.shape[0], z0s.shape[1]))
zs[0, :, :] = z0s
plt.figure()
for k in range(1, z0s.shape[1]):
	zs[:, :, k] = trajectory_parms(z0s[:, k], params_true)[1].T
	plt.plot(zs[:, 0, k], zs[:, 1, k], '.-')


def loglike_step(xkp1, xk, params):
	return 1/(2*h*params.beta**2)*(xkp1 - stepfnc_det(xk, params))**2 + np.log(sqrt(h)*params.beta)


def loglike(xs, params):
	for m in range(xs.shape[0]):
		l = 0
		for n in range(xs.shape[1]-1):
			l += loglike_step(xs[m, n+1], xs[m, n], params)
	return l




# Test des Vorwaertsmodelles
"""
ts, xs = trajectory_parms(x0s, params_true)

plt.figure();
for m, x0 in enumerate(x0s):
  plt.plot(ts, xs[m, :], label="Partikel " + str(m))
plt.legend()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Trajectories of all particles")
"""

# try inference
alphas = np.linspace(0.3, 0.7, 75)
betas = np.linspace(0.001, 0.03, 50)



# try inference
cst3 = np.zeros((75,50))
for m, alpha in enumerate(alphas):
	for n, beta in enumerate(betas):
		params = Params(alpha, beta, 0.01, 0.25, 0.001, 0.02, 0.7, 0.004)
		cst3[m, n] = loglike(xs, params)

ind = np.unravel_index(np.argmin(cst3, axis=None), cst3.shape)
alpha_gridsearch = alphas[ind[0]]
beta_gridsearch  = betas[ind[1]]
params_gridsearch =  Params(alpha_gridsearch, beta_gridsearch, 0.01, 0.25, 0.001, 0.02, 0.7, 0.004)

print("grid search: ")
print("alpha = " + str(alpha_gridsearch))
print("beta = " + str(beta_gridsearch))

plt.figure();
plt.contourf(alphas, betas, cst3.T, 50)
plt.plot(params_true.alpha, params_true.beta, '.w', markersize=10)
plt.plot(alpha_gridsearch, beta_gridsearch, '.y', markersize=10)


# optimization procedure
def optfnc(par):
	params = Params(par[0], par[1], 0.01, 0.25, 0.001, 0.02, 0.7, 0.004)
	return loglike(xs, params)

import scipy.optimize as opt

res = opt.minimize(optfnc, np.array([1.0,1.0]), method="L-BFGS-B", bounds=((None, None), (0.0001, None)), options={"disp": False})
params_opt = Params(res.x[0], res.x[1], 0.01, 0.25, 0.001, 0.02, 0.7, 0.004)


print("optimization: ")
print("alpha = " + str(res.x[0]))
print("beta = " + str(res.x[1]))

# Test des Vorwaertsmodelles
x0s = np.linspace(0, 0.6, 10)
ts, xs_opt = trajectory_parms(x0s, params_opt)

plt.figure();
for m, x0 in enumerate(x0s):
  plt.plot(ts, xs_opt[m, :], label="Partikel " + str(m))
plt.legend()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Trajectories of all particles with optimal parameters")

