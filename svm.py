import numpy as np
from lib import *



# C: regularization parameter
# tol: numerical tolerance
# kernel: kernel function
# max_pass: max times of iterate over alpha without changing
# X: training data (m, d)
# Y: training lable (m,)
def SMO(C, tol, validation_data, validation_lable, X, y, kernel = kernel_linear, max_passes=3, max_iter =100):
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
            Vrate = SMOtest(validation_data, validation_lable, X, y, alpha, b, kernel)
            print "now ir = %i; Train Correct = %r; Validation Correct = %r" %(ir, Trate, Vrate)
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

