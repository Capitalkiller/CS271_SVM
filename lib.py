import numpy as np

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

# visualize data
# e.g. visualize(train_data[1266, :], train_lable[1266])
def visualize(data, lable = None):
    data_x = data[::2]
    data_y = data[1::2]
    plt.plot(data_x, data_y)
    if lable != None:
        print "Number is : ", lable

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

def test(data, lable, model):
    # validation
    correct = 0
    for i in range(len(lable)):
        sample = data[i, :].reshape(1, -1)
        number = int(model.predict(sample))
        #print "number: ", number
        #print "lable: ", int(validation_lable[i])
        if number == lable[i]:
            correct += 1
    Correct_rate =  correct / float(len(lable))
    #print "Correct rate: ",  Correct_rate
    return Correct_rate