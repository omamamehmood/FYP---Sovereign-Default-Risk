import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.stats as stats
import bspline
import bspline.splinelab as splinelab


# EXPERIMENT 2
S0 = 100      # initial stock price
K = 80
mu = 0.05     # drift
sigma = 1.5  # volatility
r = 0.124      # risk-free rate
T = 4
M = 1         # maturity
N_MC = 3    # number of paths
delta_t = M / T                # time interval
gamma = np.exp(- r * delta_t)  # discount factor



S = pd.DataFrame([], index=range(1, N_MC+1), columns=range(T+1))

S.loc[:,0] = S0

# standard normal random numbers
RN = pd.DataFrame(np.random.randn(N_MC,T), index=range(1, N_MC+1), columns=range(1, T+1))


for t in range(1, T+1):
    S.loc[:,t] = S.loc[:,t-1] * np.exp((mu - 1/2 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * RN.loc[:,t])


delta_S = S.loc[:,1:T].values - np.exp(r * delta_t) * S.loc[:,0:T-1]

delta_S_hat = delta_S.apply(lambda x: x - np.mean(x), axis=0)

# state variable
X = - (mu - 1/2 * sigma**2) * np.arange(T+1) * delta_t + np.log(S.astype(float) / 1.0)  # delta_t here is due to their conventions


def terminal_payoff(ST, K):
    # ST   final stock price    # K    strike
    payoff = max(K - ST, 0)
    return payoff


# Define spline basis functions

X_min = np.min(np.min(X))
X_max = np.max(np.max(X))


p = 4 # order of spline (as-is; 3: cubic, 4: B-spline?)
ncolloc = 12

tau = np.linspace(X_min, X_max, ncolloc)  # These are the sites to which we would like to interpolate

# k is a knot vector that adds endpoints repeats as appropriate for a spline of order p
# To get meaninful results, one should have ncolloc >= p+1
k = splinelab.aptknt(tau, p)

# Spline basis of order p on knots k
basis = bspline.Bspline(k, p)
f = plt.figure()

print('Number of points k = ', len(k))
# basis.plot()

# Make data matrices with feature values
num_t_steps = T + 1
num_basis =  ncolloc

data_mat_t = np.zeros((num_t_steps, N_MC, num_basis))

# fill it
for i in np.arange(num_t_steps):
    x = X.values[:,i]
    basis_arr=np.array([ basis(i) for i in x ])
    data_mat_t[i,:,:] = basis_arr


# Define the risk aversion parameter
risk_lambda = 0.001 # risk aversion parameter


# functions to compute optimal hedges
def function_A_vec(t,delta_S_hat,data_mat_t,reg_param):
    # Compute the matrix A_{nm} from Eq. (52) (with a regularization!)
    X_mat = data_mat_t[t,:,:]
    num_basis_funcs = X_mat.shape[1]
    this_dS = delta_S_hat.loc[:,t].values
    hat_dS2 = (this_dS**2).reshape(-1,1)
    A_mat = np.dot(X_mat.T, X_mat * hat_dS2) + reg_param * np.eye(num_basis_funcs)
    return A_mat


def function_B_vec(t, Pi_hat, delta_S=delta_S, delta_S_hat=delta_S_hat, S=S, data_mat_t=data_mat_t,
                  gamma=gamma,risk_lambda=risk_lambda):
    coef = 1.0/(2 * gamma * risk_lambda)
    tmp =  Pi_hat.loc[:,t+1] * delta_S_hat.loc[:,t] + coef * (np.exp(mu*delta_t) - np.exp(r*delta_t))* S.loc[:,t]
    X_mat = data_mat_t[t,:,:]  # matrix of dimension N_MC x num_basis
    B = np.dot(X_mat.T, tmp)
    return B


# Compute optimal hedge and portfolio value
# portfolio value
Pi = pd.DataFrame([], index=range(1, N_MC+1), columns=range(T+1))
Pi.iloc[:,-1] = S.iloc[:,-1].apply(lambda x: terminal_payoff(x, K))

Pi_hat = pd.DataFrame([], index=range(1, N_MC+1), columns=range(T+1))
Pi_hat.iloc[:,-1] = Pi.iloc[:,-1] - np.mean(Pi.iloc[:,-1])

# optimal hedge
a = pd.DataFrame([], index=range(1, N_MC+1), columns=range(T+1))

a.iloc[:,-1] = 0

reg_param = 1e-3

for t in range(T-1, -1, -1):

    A_mat = function_A_vec(t, delta_S_hat, data_mat_t, reg_param)
    B_vec = function_B_vec(t, Pi_hat)

    # Convert A_mat and B_vec to a NumPy array of floats
    A_mat = np.array(A_mat, dtype=float)
    B_vec = np.array(B_vec, dtype=float)

    phi = np.dot(np.linalg.inv(A_mat), B_vec)
    a.loc[:,t] = np.dot(data_mat_t[t,:,:],phi)
    Pi.loc[:,t] = gamma * (Pi.loc[:,t+1] - a.loc[:,t] * delta_S.loc[:,t])
    Pi_hat.loc[:,t] = Pi.loc[:,t] - np.mean(Pi.loc[:,t])

Nd1=a.mean()

d1 = stats.zscore(Nd1)


