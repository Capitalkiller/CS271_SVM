import numpy as np
from lib import *
from svm import *
data, lable = load_train()
train_data, train_lable, validation_data, validation_lable = shuffleData(data, lable)
print "train data: ", train_data.shape
print "train lable: ", train_lable.shape
train_lable[train_lable <= 4 ] = -1
train_lable[train_lable > 4] = 1

alpha, b = SMO(2, 1e-4, train_data, train_lable, max_passes=3)
print alpha.max()