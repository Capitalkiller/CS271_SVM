import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import svm
from lib import *
from svm import *

#load data
train_data, train_lable =  load_train("./data/0vs8Source.csv")
print "train data: ", train_data.shape
print "train lable: ", train_lable.shape
target_data, target_lable  = load_train("./data/0vs8Target.csv")
print "target data: ", target_data.shape
print "target lable: ", target_lable.shape
test_data  = load_test("./data/0vs8TestNoLabels.csv")
print "train data: ", test_data.shape


# it is 0 vs 8 problem, so divde only two class
train_lable[train_lable <= 4 ] = -1
train_lable[train_lable > 4] = 1
target_lable[target_lable <= 4 ] = -1
target_lable[target_lable > 4] = 1

# train on train data
w, b = QP_Slover(train_data, train_lable)
# transfer learning on target data
k = 0.0006
w_target, b_target = QP_Slover_transfer(target_data, target_lable, w, k)

#predict
pre_test = prediction_QP(w_target, b_target, test_data)
pre_test[pre_test>0] = 8
pre_test[pre_test<0] = 0
np.savetxt('predect_0vs8_test.txt', pre_test,fmt='%i')