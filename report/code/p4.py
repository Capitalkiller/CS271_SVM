import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from lib import *
from svm import *

#loaddata
train_data, train_lable =  load_train("./data/pendigits-train.csv")
print "train data: ", train_data.shape
print "train lable: ", train_lable.shape
test_data  = load_test("./data/pendigits-test-nolabels.csv")
print "test data: ", test_data.shape

#train
train_lable_num, alpha_num, b_num = train_0va(train_data, train_lable, max_passes=10, kernel=kernel_gaussian, max_iter=3,
                                             validation_data = validation_data, validation_lable = validation_lable )

# predict
number = np.zeros(len(test_data))
for i in range(len(test_data)):
    number[i] = int(number_predict(test_data[i, :], train_data, train_lable_num, alpha_num, b_num, kernel_gaussian))
number = number.astype(int)
np.savetxt('prediction.txt', number,fmt='%i')