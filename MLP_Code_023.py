import numpy as np
import matplotlib.pyplot as plt
npoints = 50
X, Y = [], []
#class0
X.append(np.random.uniform(low = -2.5, high = 2.3, size = (npoints, )))
Y.append(np.random.uniform(low = -1.7, high = 2.8, size = (npoints, )))
#class1
X.append(np.random.uniform(low = -7.2, high = -4.4, size = (npoints, )))
Y.append(np.random.uniform(low = 3, high = 6.5, size = (npoints, )))
learnset = []
learnlabels = []
for i in range(2):
    points = zip(X[i], Y[i])

    for p in points:
        learnset.append(p)
        learnlabels.append(i)

npoints_test = 3*npoints

TestX = np.random.uniform(low = -7.2, high = 5, size = (npoints_test, ))
TestY = np.random.uniform(low = -4, high = 9, size = (npoints_test, ))

test_set = []

points = zip(TestX, TestY)

for p in points:
    test_set.append(p)

colors = ["b", "r"]

for i in range(2):
    plt.scatter(X[i], Y[i], c = colors[i])

plt.scatter(TestX, TestY, c = "g")
plt.show()
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes= (20, 3), max_iter= 150, alpha= 1e-4, solver= 'sgd', verbose= 10, tol= 1e-4, random_state= 1, learning_rate_init= 0.1)
mlp.fit(learnset, learnlabels)

print("Training set score: %f" %mlp.score(learnset, learnlabels))
print("Test set score: %f" %mlp.score(learnset, learnlabels))

mlp.classes_
predictions = mlp.predict(test_set)
predictions
testset = np.array(test_set)
testset[predictions == 1]
colors = ['#C0FFFF', '#FFC8C8']

for i in range(2):
    plt.scatter(X[i], Y[i], c = colors[i])

colors = ["r", "g"]

for i in range(2):
    cls = testset[predictions == i]
    Xt, Yt = zip(*cls)
plt.scatter(Xt, Yt, c = colors[i])
