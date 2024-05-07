import numpy as np
from PIL import Image
import math
import statistics as stat
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
import json
from preprocessing import *
from housekeeping import *
import matplotlib.pyplot as plt


def train_test(X, y, k=5, verbose=False, plot=False, store=False):
    folds = kfold(len(X), k)
    depths =  [None, 1, 2, 3, 4, 5]  


    results = [] 
    for max_d in depths:
        temp = []
        for i, fold in enumerate(folds):
            train_i = []
            for j in range(len(folds)):
                if j!=i:
                    train_i.extend(folds[j])

            X_train = X[train_i]
            y_train = y[train_i]
            X_test = X[folds[i]]
            y_test = y[folds[i]]

            clf = RandomForestClassifier(max_depth=max_d)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            acc, prec, rec, f1 = metrics(y_pred, y_test)
            temp.append((acc, prec, rec, f1))

        avg_acc = sum([x[0] for x in temp])/k
        avg_prec = sum([x[1] for x in temp])/k
        avg_rec = sum([x[2] for x in temp])/k
        avg_f1 = sum([x[3] for x in temp])/k

        if verbose:
            print(f"Max Depth = {max_d}")
            print(f"average accuracy {avg_acc}")
            print(f"average precision {avg_prec}")
            print(f"average recall {avg_rec}")
            print(f"average f1 {avg_f1}")
            print()

        results.append((avg_acc, avg_prec, avg_rec, avg_f1))

    if plot:
        depths= [0,1,2,3,4,5]
        acc = [x[0] for x in results]
        prec = [x[1] for x in results]
        rec = [x[2] for x in results]
        f1 = [x[3] for x in results]
        plt.plot(depths, acc,  label= 'accuracy')
        plt.plot(depths, prec, label='precision')
        plt.plot(depths, rec, label='recall')
        plt.plot(depths, f1, label='f1')
        plt.title('Metrics w.r.t. Max Depth')
        plt.xlabel("Max Depth")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

    if store:
        ls = []
        for depth, res in zip(depths, results):
            ls.append({
                "max depth": depth,
                "avg accuracy": res[0],
                "avg precision": res[1],
                "average recall": res[2],
                "average f1": res[3]
            })

        with open("results/tf_results_by_max_depth.json", "w") as f:
            json.dump(ls, f)



if __name__ == "__main__":
    X, y = data_prep(lda=True, norm=False)
    results = train_test(X, y, verbose=True, plot=True)