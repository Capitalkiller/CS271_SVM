import numpy as np
import matplotlib.pyplot as plt



# svm implementation
# no kernel
# xi: (m, d)
# x: (d,)
# return (m,)
def kernel_linear(xi, x):
    inner_product = np.dot(xi, x)
    inner_product = inner_product.T
    # print inner_product.shape
    return inner_product

def kernel_gaussian(xi, x, sigma = 40):
    # print "xi",xi.shape
    # print "x",x.shape
    if len(xi) == len(x):
        similarity = np.exp(-np.linalg.norm(xi - x)**2/(2*sigma**2))
    else:
        similarity = np.exp(-np.linalg.norm((xi - x), axis=1)**2/(2*sigma**2))
    return similarity

def kernel_polynomial(xi, x, d = 15, c = 18):
    phi_xz = (np.dot(xi,x) + c)**d
    phi_xz = phi_xz.T

    return phi_xz

def predict(test, X, y, alpha, b, kernel = kernel_linear):
    value = (alpha * y * kernel(X, test)).sum() + b
    y = np.round(value)
    return y

# load train data
def load_train():
    train = np.loadtxt("./data/pendigits-train.csv",delimiter=',')
    train.shape
    train_data = train[:, :16]
    train_lable = train[:, 16:17]
    train_lable = train_lable.reshape(len(train_lable))
    return train_data, train_lable

def load_test():
    test = np.loadtxt("./data/pendigits-test-nolabels.csv", delimiter=',')
    test = test.reshape(len(test), -1)
    print test.shape
    return



def shuffleData(data, lable):
    data_index = np.arange(len(data))
    np.random.shuffle(data_index)
    train_index = data_index[:3000]
    validation_index = data_index[3000:]
    train_data = data[train_index]
    train_lable = lable[train_index]
    validation_data = data[validation_index]
    validation_lable = lable[validation_index]
    return train_data, train_lable, validation_data, validation_lable

# def test(data, lable, model):
#     # validation
#     correct = 0
#     for i in range(len(lable)):
#         sample = data[i, :].reshape(1, -1)
#         number = int(model.predict(sample))
#         #print "number: ", number
#         #print "lable: ", int(validation_lable[i])
#         if number == lable[i]:
#             correct += 1
#     Correct_rate =  correct / float(len(lable))
#     #print "Correct rate: ",  Correct_rate
#     return Correct_rate

# predict number using 0va model
def number_predict(test, train_data, train_lable_num, alpha_num, b_num, kernel):
    score_num = np.zeros(10)
    for i in range(10):
        score_num[i] = predict(test, train_data, train_lable_num[i], alpha_num[i], b_num[i], kernel)
    number = int(np.argmax(score_num))
    return number

# validation error rate
def validation_correct_rate(validation_data, validation_lable, train_data, train_lable_num, alpha_num, b_num, kernel_gaussian):
    correct = 0
    for i in range(len(validation_lable)):
        number = number_predict(validation_data[i, :], train_data, train_lable_num, alpha_num, b_num, kernel_gaussian)
#         print "%i sample, number = %i, lable = %i" %(i, number, validation_lable[i])
        if number == int(validation_lable[i]):
            correct += 1
    correct_rate = float(correct) / len(validation_lable)
    return correct_rate

def SMOtest(validation, validation_lable, data, lable, alpha, b, kernel = kernel_linear):
    # validation
    correct = 0
    for i in range(len(validation_lable)):
        sample = validation[i, :]
        pre = predict(sample, data, lable, alpha, b, kernel)
        if pre < 0:
            pre = -1
        else:
            pre = 1
        #print "number: ", number
        #print "lable: ", int(validation_lable[i])
        if pre == validation_lable[i]:
            correct += 1
    Correct_rate =  correct / float(len(validation_lable))
    #print "Correct rate: ",  Correct_rate
    return Correct_rate