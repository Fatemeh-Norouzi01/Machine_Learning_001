import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target
print(iris_data)
print(iris_labels)
np.random.seed(42)

indices = np.random.permutation(len(iris_data))
print(indices)
n_training_samples = 12

learnset_data = iris_data[indices[:-n_training_samples]]
learnset_labels = iris_labels[indices[:-n_training_samples]]
print(learnset_data)
print(learnset_labels)
testset_data = iris_data[indices[-n_training_samples:]]
testset_labels = iris_labels[indices[-n_training_samples:]]
print(testset_data)
print(testset_labels)
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X = []

for iclass in range(3):
    X.append([[], [], []])

    for i in range(len(learnset_data)):
        if learnset_labels[i] == iclass:

            X[iclass][0].append(learnset_data[i][0])
            X[iclass][1].append(learnset_data[i][1])
            X[iclass][2].append(sum(learnset_data[i][2:]))

colours = ("r", "g", "y")

fig = plt.figure()

ax = fig.add_subplot(111, projection= '3d')

for iclass in range(3):
    ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c = colours[iclass])

plt.show()
def distance(instance1, instance2):
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1- instance2)

def get_neighbors(training_set, labels, test_instance, k, distance = distance ):
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key = lambda x: x[1])
    neighbors = distances[:k]

    return neighbors
for i in range(5):
    neighbors = get_neighbors(learnset_data, learnset_labels, testset_data[i], 3, distance= distance)

print(i, testset_data[i], testset_labels[i], neighbors)
from collections import Counter

def Vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    return class_counter.most_common(1)[0][0]
for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data, learnset_labels, testset_data[i], 3, distance= distance)
    print("index: ", i, ", result of vote: ", Vote(neighbors), ", label: " ,testset_labels[i], ", data: ", testset_data[i])
# Create and fit a nearest-neighbor classifier

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(learnset_data, learnset_labels)
KNeighborsClassifier(algorithm= 'auto', leaf_size= 30, metric= 'minkowski', metric_params= None, n_jobs= 1, n_neighbors= 5, p= 2, weights= 'uniform')

print("Predictions from the classifier: ")
print(knn.predict(testset_data))
print("Target values: ")
print(testset_labels)
