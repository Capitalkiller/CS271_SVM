import numpy as np
from lib import *
from scipy.optimize import minimize



# C: regularization parameter
# tol: numerical tolerance
# kernel: kernel function
# max_pass: max times of iterate over alpha without changing
# X: training data (m, d)
# Y: training lable (m,)
def SMO(C, tol, X, y, kernel = kernel_linear, max_passes=3, max_iter =100, validation_data = None, validation_lable = None):
    m = len(y) # number of samples
    alpha = np.zeros(m)
    b = 0
    passes = 0
    ir = 0
    Error = np.zeros(m)   

    while passes < max_passes and ir < max_iter:
        ir += 1
        if ir % 1 == 0:
            Trate = SMOtest(X, y, X, y, alpha, b, kernel)
            if validation_lable != None:
                Vrate = SMOtest(validation_data, validation_lable, X, y, alpha, b, kernel)
                print "now ir = %i; Train Correct = %r, validation Correct = %r; " %(ir, Trate, Vrate)
            else:
                print "now ir = %i; Train Correct = %r; " %(ir, Trate)

        num_changed_alpha = 0
        for i in range(m):
            Error[i] = (alpha * y * kernel(X, X[i])).sum() + b - y[i] # f(xi) - yi
            if(((y[i] * Error[i] < -tol) and (alpha[i] < C)) or ((y[i] * Error[i] > tol) and (alpha[i] > 0))):

                j = np.random.randint(0, m)
                while j == i:
                    j = np.random.randint(0, m)
                Error[j] = (alpha * y * kernel(X, X[j])).sum() + b - y[j] # f(xj) - yj
                alpha_old_i = alpha[i]
                alpha_old_j = alpha[j]

                if y[i] != y[j]:
                    L = np.maximum(0, alpha[j] - alpha[i])
                    H = np.minimum(C, C + alpha[j] - alpha[i])
                else:
                    L = np.maximum(0, alpha[i] + alpha[j] - C)
                    H = np.minimum(C, alpha[i] + alpha[j])

                # if L > H:
                #     print "H", H
                #     print "L", L

                if L == H:
                    continue
                eta = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j])
                
                if eta >= 0:
                    continue

                alpha[j] = alpha[j] - y[j] * (Error[i] - Error[j]) / eta
                # clip aj
                if alpha[j] > H:
                    alpha[j] = H
                if alpha[j] < L:
                    alpha[j] = L



                if np.abs(alpha[j] - alpha_old_j) < 1e-5:
                    continue

                alpha[i] = alpha[i] + y[i] * y[j] * (alpha_old_j - alpha[j])
                # print "j: ", alpha[j]
                # print "i: ", alpha[i]

                b1 = b - Error[i] - y[i] * (alpha[i] - alpha_old_i) * (kernel(X[i], X[i])) - y[i] * (alpha[j] - alpha_old_j) * (kernel(X[i], X[j]))
                b2 = b - Error[i] - y[i] * (alpha[i] - alpha_old_i) * (kernel(X[i], X[j])) - y[i] * (alpha[j] - alpha_old_j) * (kernel(X[j], X[j]))

                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2
                num_changed_alpha += 1

        if num_changed_alpha == 0:
            passes += 1
        else:
            passes = 0

    return alpha, b

# train 9 0va model to do 0va prediction    
def train_0va(train_data, train_lable, max_passes=3, kernel=kernel_gaussian, max_iter=3, validation_data = None, validation_lable = None):
    train_lable_num = []
    alpha_num = []
    b_num = []
    validation_lable_num = []
    validation_lable_one = None
    for i in range (10):        
        lable_one = np.copy(train_lable)
        lable_one[train_lable == i] = 1
        lable_one[train_lable != i] = -1    
        train_lable_num.append(lable_one)
        if validation_lable != None:            
            validation_lable_one = np.copy(validation_lable)
            validation_lable_one[validation_lable == i] = 1
            validation_lable_one[validation_lable != i] = -1    
            validation_lable_num.append(validation_lable_one)

        print "Now doing number: %i, there is %i of them" %(i, (train_lable == i).sum())
        alpha_one, b_one = SMO(2, 1e-5,  train_data, lable_one, kernel, max_passes, max_iter, validation_data =validation_data, validation_lable = validation_lable_one )
        alpha_num.append(alpha_one)
        b_num.append(b_one)
    return train_lable_num, alpha_num, b_num

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
    
def crossvalidation(c_lb,c_ub,d_lb,d_ub,train_data, train_lable, validation_data, validation_lable):
    hyperpara_pairs = np.mgrid[c_lb:c_ub:1, d_lb:d_ub:1].reshape(2,-1).T
# validation process to find optimized c and d
    correct_rate_max = 0
    num_pairs = len(hyperpara_pairs)
    for i in np.arange(num_pairs):
        ci = hyperpara_pairs[i][0] 
        di = hyperpara_pairs[i][1]
        # training with polynominal kernal
        kernel_poly = partial(kernel_polynomial, c = ci,d = di)
        train_lable_num, alpha_num,b_num = train_0va(train_data, train_lable, max_passes=3, kernel=kernel_poly, max_iter=3, 
            validation_data =validation_data, validation_lable = validation_lable)
        # predict the correction
        correct_rate = validation_correct_rate(validation_data, validation_lable, 
            train_data, train_lable_num, alpha_num, b_num, kernel_poly)
        print i, "/", num_pairs, "correction rate is ", correct_rate 
        # judege the best one?
        if correct_rate > correct_rate_max:
            correct_rate_max = correct_rate
            c_best = ci
            d_best = di

    print "The best correct rate is ", correct_rate_max
    print "The hyperpameter are c = ", c_best, "d = ", d_best

    return correct_rate_max, c_best, d_best