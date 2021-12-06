import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from itertools import cycle, islice



def get_serial(input_file):
    out_file = os.path.splitext(input_file)[0] + "_serial.txt"
    y = []
    with open(out_file) as f:
        num_points = int(f.readline().strip())
        for _ in range(num_points):
            info = f.readline().strip()
            y.append(int(info))
    return y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark task runner")
    parser.add_argument("file", type=str, help="json file of tasks")
    

    args = parser.parse_args()

    y_serail = get_serial(args.file)
    
    x = []
    with open(args.file) as f:
        num_points = int(f.readline().strip())
        info = f.readline().strip().split()
        eps, min_samples = float(info[0]), int(info[1])
        for _ in range(num_points):
            info = f.readline().strip().split()
            x.append([float(info[0]), float(info[1])])
    X = np.asarray(x)
    dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = dbscan.fit_predict(X)
    colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred+y_serail) + 1),
                )
            )
        )
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])

    fig, axs = plt.subplots(2)
    fig.suptitle(args.file)
    axs[0].scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])



    axs[1].scatter(X[:, 0], X[:, 1], s=10, color=colors[y_serail])
    plt.savefig(os.path.splitext(args.file)[0]+".png")


        


