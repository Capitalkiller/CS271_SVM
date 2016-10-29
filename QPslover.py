import numpy as np
from lib import *
from scipy.optimize import minimize

def QP_Slover(train_data, train_lable):
    # maxmize W(a)
    def max_w(x):
        return  np.linalg.norm(x[:16])
    # constrains:
    cons = ({'type': 'ineq',
     'fun' : lambda x: train_lable * (np.dot(x[:16], train_data.T) + x[16]) - 1})
    # initialize W
    w_init = np.random.rand(17)
    # optimization
    res = minimize(max_w, w_init , method='SLSQP', constraints=cons, options={'disp': True})
    w_pre = res.x[:16]
    b_pre = res.x[16]
    return w_pre, b_pre

def QP_Slover_transfer(train_data, train_lable, w_previous, epsilon):
    # maxmize W(a)
    def max_w(x):
        return  np.linalg.norm(x[:16])
    # constrains:
    cons = ({'type': 'ineq',
     'fun' : lambda x: train_lable * (np.dot(x[:16], train_data.T) + x[16]) - 1},
	{'type': 'ineq',
     'fun' : lambda x: -(np.linalg.norm(w_previous - x[:16]) - epsilon)})
    # initialize W
    w_init = np.random.rand(17)
    # optimization
    res = minimize(max_w, w_init , method='SLSQP', constraints=cons, options={'disp': True})
    w_pre = res.x[:16]
    b_pre = res.x[16]
    return w_pre, b_pre

def prediction_QP(w, b, test_data):
    pre_validation = np.dot(w, test_data.T) + b
    pre_validation[pre_validation >= 0] =1
    pre_validation[pre_validation <0 ] = -1
    return pre_validation