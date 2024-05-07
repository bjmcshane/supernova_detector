from tkinter import W
import numpy as np
from PIL import Image
import math
import statistics as stat
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
import json
from preprocessing import *
from housekeeping import *
import matplotlib.pyplot as plt


def train_test(X, y, k=5, verbose=False, plot=False, store=False):
    folds = kfold(len(X), k) 
    cs = np.linspace(.1,2,num=11)

    results = []
    for c in cs:
        lr = LogisticRegression(C=c)

        temp = []
        for i, fold in enumerate(folds):
            test_i = fold
            train_i = []
            for j in range(len(folds)):
                if j!=i:
                    train_i.extend(folds[j])

            X_train = X[train_i]
            y_train = y[train_i]
            X_test = X[test_i]
            y_test = y[test_i]


            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            acc, prec, rec, f1 = metrics(y_pred, y_test)
            temp.append((acc, prec, rec, f1))

        avg_acc = sum([x[0] for x in temp])/k
        avg_prec = sum([x[1] for x in temp])/k
        avg_rec = sum([x[2] for x in temp])/k
        avg_f1 = sum([x[3] for x in temp])/k

        if verbose:
            print(f"C = {c}")
            print(f"average accuracy {avg_acc}")
            print(f"average precision {avg_prec}")
            print(f"average recall {avg_rec}")
            print(f"average f1 {avg_f1}")
            print()
        
        results.append((avg_acc, avg_prec, avg_rec, avg_f1))

    if plot:
        acc = [x[0] for x in results]
        prec = [x[1] for x in results]
        rec = [x[2] for x in results]
        f1 = [x[3] for x in results]
        plt.plot(cs, acc,  label= 'accuracy')
        plt.plot(cs, prec, label='precision')
        plt.plot(cs, rec, label='recall')
        plt.plot(cs, f1, label='f1')
        plt.title("Metrics w.r.t. Regularization Parameter C")
        plt.xlabel("Regularization Parameter C")
        plt.ylabel("Score")
        plt.legend()
        plt.show()


    if store:
        ls = []
        for c, res in zip(cs, results):
            ls.append({
                "C": c,
                "avg accuracy": res[0],
                "avg accuracy": res[1],
                "average recall": res[2],
                "average f1": res[3]
            })

        with open("results/lr_results_by_c.json", "w") as f:
            json.dump(ls, f)


if __name__ == "__main__":
    X, y = data_prep(lda=True, norm=True)
    results = train_test(X, y, plot=True)