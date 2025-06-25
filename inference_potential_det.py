import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from math import sqrt

# Domäne: x \in [0,1], t \in [0,1]

# Dynamik: Vektorfeld u(t, x)
# Zellen: x_i' = alpha*u(t, x_i(t)) + grad(Phi(t, x_i(t)))

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


params_true = Params(0.5, 0.02, 0.01, 0.25, 0.001, 0.02, 0.7, 0.004)

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
	return 1/(2*h*params.beta**2)*(xkp1 - stepfnc_det(xk, params))**2  + np.log(sqrt(h)*params.beta)

def loglike(xs, params):
	l = 0
	for m in range(xs.shape[0]):
		for n in range(xs.shape[1]-1):
			l += loglike_step(xs[m, n+1], xs[m, n], params)
	return l

def loglike_arr(xs, params):
	l = np.zeros((xs.shape[0], xs.shape[1]-1))
	for m in range(xs.shape[0]):
		for n in range(xs.shape[1]-1):
			l[m, n] = loglike_step(xs[m, n+1], xs[m, n], params)
	return l

xvals = np.linspace(0, 1, 100)

x0s = np.linspace(0, 0.6, 10)
ts, xs = trajectory(x0s, params_true)
plt.figure(); plt.ion()
plt.plot(xvals, params_true.Phi(xvals))
#plt.plot(xvals, GradPhi(xvals))
plt.title("Potential $\Phi$")
plt.show()

params_ = Params(0.5, 0.02, 0.01, 0.2, 0.001, 0.02, 0.7, 0.004) # perturbed parameters
l1 = loglike_arr(xs, params_true)
l2 = loglike_arr(xs, params_)


# try inference
N1 = 30
N2 = 30
posa = np.linspace(-0.5, 1.5, N1)
posb = np.linspace(-0.5, 1.5, N2)



# try inference
cst3 = np.zeros((N1, N2))
for m, pa in enumerate(posa):
	for n, pb in enumerate(posb):
		params = Params(0.5, 0.02, 0.01, pa, 0.001, 0.02, pb, 0.004)
		cst3[m, n] = loglike(xs, params)

ind = np.unravel_index(np.argmin(cst3, axis=None), cst3.shape)
posa_gridsearch = posa[ind[0]]
posb_gridsearch  = posb[ind[1]]
params_gridsearch =  Params(0.5, 0.02, 0.01, posa_gridsearch, 0.001, 0.02, posb_gridsearch, 0.004)

print("grid search: ")
print("pos_a = " + str(posa_gridsearch))
print("pos_b = " + str(posb_gridsearch))

plt.figure(2);
plt.contourf(posa, posb, cst3.T, 50)
plt.plot(params_true.pos_a, params_true.pos_b, '.w', markersize=20, label="true")
plt.plot(posa_gridsearch, posb_gridsearch, '.y', markersize=20, label="grid search")


"""import time
s1 = time.time()
optfnc([0.5, 0.25, 0.7])
s2 = time.time()
loglike(xs, params_true)
s3 = time.time()
print(s3-s2)
print(s2-s1)"""

#print(loglike_variant(xs, params_, x0s))

# (0.5, 0.02, 0.01, 0.25, 0.001, 0.02, 0.7, 0.004)
import scipy.optimize as opt
def optfnc(par):
	params = Params(par[0],par[1],  par[2], par[3], par[4], par[5], par[6], par[7])
	return loglike(xs, params)
res = opt.minimize(optfnc, np.array([0.3, 0.01, 0.05, 0.2, 0.002, 0.01, 0.85, 0.003]), method="L-BFGS-B", bounds=((None, None),(0.00001, None),(None,None),(None, None),(None, None),(None,None),(None, None),(None,None)), options={"disp": True})
#res = opt.minimize(optfnc, np.array([0.3, 0.01, 0.2, 0.85]), method="L-BFGS-B", bounds=((None, None),(0.00001, None),(None,None),(None, None),(None,None),(None, None),(None,None),(None, None)), options={"disp": True})
params_opt = Params(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.x[6], res.x[7])

plt.plot(res.x[3], res.x[6], '.m', markersize=20, label="opt search")
plt.legend()


# Test des Vorwaertsmodelles
ts, xs_ = trajectory(x0s, params_opt)
ts, xs_2 = trajectory(x0s, params_gridsearch)

plt.figure(3); 
for m, x0 in enumerate(x0s):
  plt.plot(ts, xs[m, :], label="Partikel " + str(m))
  
  
plt.title("true params")
plt.legend()
plt.savefig("true.png")

plt.figure(4);
for m, x0 in enumerate(x0s):
  plt.plot(ts, xs_[m, :], label="Partikel opt " + str(m))



plt.title("optimized params")
plt.legend()
plt.savefig("opt.png")


plt.figure(); plt.ion()
plt.plot(xvals, params_opt.Phi(xvals))
#plt.plot(xvals, GradPhi(xvals))
plt.title("Potential $\Phi$ (opt)")

plt.figure(5);
for m, x0 in enumerate(x0s):
  plt.plot(ts, xs_2[m, :], label="Partikel" + str(m))
plt.title("grid search")
plt.legend()
plt.savefig("grid.png")


plt.figure(2)
plt.xlabel("pos_a")
plt.ylabel("pos_b")
plt.savefig("error")

