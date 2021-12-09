import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from itertools import cycle, islice



def get_cluster(input_file, suffix):
    out_file = os.path.splitext(input_file)[0] + suffix
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
    # parser.add_argument("file", type=str, help="json file of tasks")
    suffixs = ["_ds-seq.txt", "_ds-shm.txt", "_gdbscan.txt", "_serial.txt", "_hybrid.txt"]
    filenum = 6
    nums = len(suffixs) + 1 # 1 for sklearn's implementation
    fig, axs = plt.subplots(nums, filenum, sharex=True, sharey=True)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle("Validation")
    for fileId in range(6):
        file_path = "../data/validation/simple_{}_1500.txt".format(fileId)

        ys = []
        for suffix in suffixs:
            ys.append(get_cluster(file_path, suffix))
        
        x = []

        with open(file_path) as f:
            num_points = int(f.readline().strip())
            info = f.readline().strip().split()
            eps, min_samples = float(info[0]), int(info[1])
            for _ in range(num_points):
                info = f.readline().strip().split()
                x.append([float(info[0]), float(info[1])])
        X = np.asarray(x)
        dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X)
        max_y = max(y_pred)
        for y in ys:
            max_y = max(max_y, max(y))
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
                        int(max_y + 1),
                    )
                )
            )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])

        
        axs[0, fileId].scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
        


        for i in range(len(suffixs)):
            axs[i+1, fileId].scatter(X[:, 0], X[:, 1], s=10, color=colors[ys[i]])
            
        # break # FIXME:    
    axs[0, 0].set_title('sklearn.DBSCAN')
    for i, suffix in enumerate(suffixs):
        axs[i+1, 0].set_title(suffix.split('.')[0][1:])


    
    
    plt.savefig("validation.png")


        


