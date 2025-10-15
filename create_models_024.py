Uniform function
import numpy as np
import matplotlib.pyplot as plt
number_of_samples = 1200

low = -1
high = 0
s = np.random.uniform(low, high, number_of_samples)
plt.hist(s)
plt.show()
______________________________________________________________________________________________________
Binomial
n = 100
p = 0.5
size = 1200
s = np.random.binomial(n, p, size)
plt.hist(s)
plt.show()
________________________________________________
Truncnorm
from scipy.stats import truncnorm
def truncate_normal(mean = 0, sd = 1, low = 0, upp = 10):
    return truncnorm((low-mean)/ sd,
                     (upp-mean)/ sd,
                     loc = mean,
                     scale = sd)
X = truncate_normal(
    mean = 0,
    sd = 0.4,
    low = -0.5,
    upp = 0.5
)

s = X.rvs(10000)
plt.hist(s)
plt.show()
_______________________
Confusion Matrix
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import pandas as pd
image_size = 28
no_different_labels = 10
image_pixels = image_size * image_size
train_df = pd.read_csv("data/mnist_train.csv")
test_df = pd.read_csv("data/mnist_test.csv")
fac = 0.99/255

train_imgs = np.asarray(train_df.iloc[:, 1:], dtype=float) * fac + 0.01
test_imgs  = np.asarray(test_df.iloc[:, 1:], dtype=float)  * fac + 0.01

train_labels = np.asarray(train_df.iloc[:, 0], dtype=int)
test_labels  = np.asarray(test_df.iloc[:, 0], dtype=int)
lr = np.arange(10)

for label in range(10):
    one_hot = (lr == label).astype(int)
    print("label=", label, " in one-hot representation: ", one_hot)
lr = np.arange(no_different_labels)

train_labels_one_hot = (lr == train_labels[:, None]).astype(float)
test_labels_one_hot  = (lr == test_labels[:, None]).astype(float)
train_labels_one_hot[train_labels_one_hot == 0] = 0.1
test_labels_one_hot[test_labels_one_hot == 0]   = 0.1
train_labels_one_hot[train_labels_one_hot == 1] = 0.99
test_labels_one_hot[test_labels_one_hot == 1] = 0.99
import pickle
import os
os.makedirs("data", exist_ok=True)
path = "data/mnist_data.pkl"

with open(path, "wb") as fh:
    data = (
        train_imgs,
        test_imgs,
        train_labels,
        test_labels_one_hot,
        train_labels_one_hot
    )
    pickle.dump(data, fh)

print("Pickle saved at:", path)
with open(path, "rb") as fh:
    data = pickle.load(fh)
    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2]
    test_labels_one_hot = data[3]
    train_labels_one_hot = data[4]

image_size = 28
no_of_different_labels = 10
image_pixels = image_size * image_size
print("train_imgs shape:", train_imgs.shape)
print("test_imgs shape:", test_imgs.shape)
print("train_labels shape:", train_labels.shape)
print("test_labels shape:", test_labels.shape)
print("train_labels_one_hot shape:", train_labels_one_hot.shape)
print("test_labels_one_hot shape:", test_labels_one_hot.shape)
@np.vectorize

def sigmoid(x):
    return 1/ (1+ np.e** -x)

activation_function = sigmoid

def truncated_normal(mean = 0, sd = 1, low = 0, upp = 10):
    return truncnorm((low-mean)/sd, (upp-mean)/sd, loc = mean, scale = sd)
class NeuralNetwork:
    
    def __init__(self,
                no_in_nodes,
                no_out_nodes,
                no_hidden_nodes,
                learning_rate):
        self.no_in_nodes = no_in_nodes
        self.no_out_nodes = no_out_nodes
        self.no_hidden_nodes = no_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        rad = 1/ np.sqrt(self.no_in_nodes)
        X  = truncated_normal(mean = 0, sd = 1, low = -rad, upp = rad)
        
        self.wih = X.rvs((self.no_hidden_nodes, self.no_in_nodes))
        
        rad = 1/ np.sqrt(self.no_hidden_nodes)
        X  = truncated_normal(mean = 0, sd = 1, low = -rad, upp = rad)
    
        self.who = X.rvs((self.no_out_nodes, self.no_hidden_nodes))

    def train(self, input_vector, target_vector):
        input_vector = np.array(input_vector, ndmin =2).T
        target_vector = np.array(target_vector, ndmin =2).T
        
        output_vector1 = np.dot(self.wih, input_vector)
    
        output_hidden = activation_function(output_vector1)
        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)
        output_error = target_vector - output_network
        
        #update the weights:
        tmp = output_error * output_network * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)
        self.who += tmp
    
        #calculate hidden errors:
        hidden_errors = np.dot(self.who.T, output_error)
    
        #update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        self.wih += self.learning_rate * np.dot(tmp, input_vector.T)

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin = 2).T
        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        output_vector = np.dot(self.who, output_vector)
        oytput_vector = activation_function(output_vector)
    
        return output_vector

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = int(labels[i])
            cm[res_max, int(target)] += 1
    
        return cm

    def precision(self, label, confusion_matrix):
        row = confusion_matrix[:, label]
        return confusion_matrix[label, label] / row.sum()

    def recall(self, label, confusion_matrix):
        column = confusion_matrix[label, :]
        return confusion_matrix[label, label] / column.sum()

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs
ANN = NeuralNetwork(no_in_nodes = image_pixels,
                   no_out_nodes = 10,
                   no_hidden_nodes = 100,
                   learning_rate = 0.1)
for i in range(len(train_imgs)):
    ANN.train(train_imgs[i], train_labels_one_hot[i])
for i in range(20):
    res = ANN.run(test_imgs[i])
    print(test_labels[i], np.argmax(res), np.max(res))
corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
print("accuracy of  train:", corrects/(corrects + wrongs))

corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
print("accuracy of  test:", corrects/(corrects + wrongs))

cm = ANN.confusion_matrix(train_imgs, train_labels)
print(cm)

for i in range(10):
    print("digit", i, "percision:", ANN.precision(i, cm), "recall: ", ANN.recall(i, cm))
