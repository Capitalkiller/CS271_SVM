import numpy as np
from lib import *
from svm import *
from functools import partial
data, lable = load_train()
train_data, train_lable, validation_data, validation_lable = shuffleData(data, lable)
print "train data: ", train_data.shape
print "train lable: ", train_lable.shape
# train_lable[train_lable == 1 ] = -1
# train_lable[train_lable != 1] = 1
# kernel_poly = partial(kernel_polynomial, c = 13,d = 13)
# fit training one number each
# lable_one = np.copy(train_lable)
# lable_one[train_lable == 9] = 1
# lable_one[train_lable != 9] = -1    
# alpha_one, b_one = SMO(2, 1e-5,  train_data, lable_one, kernel_poly, max_passes=7, max_iter=7)

# train 0va
# train_lable_num, alpha_num, b_num = train_0va(train_data, train_lable, max_passes=5, kernel=kernel_poly, max_iter=5)
# correct_rate = validation_correct_rate(validation_data, validation_lable, train_data, train_lable_num, alpha_num, b_num, kernel_poly)
# print correct_rate

# # adjusting hyperparameter with validation set 
# # set the bounds of c: 1-100
c_lb = 10
c_ub = 22
# set the bounds of d: 1-10
d_lb = 9
d_ub = 20

correct_rate_max, c_best, d_best = crossvalidation(c_lb,c_ub,d_lb,d_ub,train_data, train_lable, validation_data, validation_lable)