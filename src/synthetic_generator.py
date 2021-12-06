import numpy
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

paramters = [
    [(0, 10), 5, 10000, 1]
]

colors = list("bgrcmyk")
for param in paramters:
    center_box, centers, n_samples, cluster_std = param
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=2, random_state=0, cluster_std=cluster_std, center_box=center_box)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for x, y in zip(X, y):
        plt.plot(x[0], x[1], colors[y] + ".")
    plt.savefig('test.png')
