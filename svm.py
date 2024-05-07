
import numpy as np
from PIL import Image
import math
import statistics as stat
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
import json
from preprocessing import *
from housekeeping import *
import time





def grid_search(X, y, verbose=False, store=False):
    print(time.time())
    param_grid = {
        "kernel" : ['poly', 'rbf'],
        "gamma" : [.01, .1, 1],
        "C": [.1, 1, 10]
    }


    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3, cv=5)
    grid.fit(X, y)

    results = {
        'best_params': grid.best_params_,
        'best_estimator': grid.best_estimator_,
        'best_score': grid.best_score_
    }

    if verbose:
        print('best params')
        print(results['best_params'])
        print('best score')
        print(results["best_score"])
        print('best estimator')
        print(results['best_estimator'])
    
    if store:
        with open("results/best_svm.json", "w") as f:
            json.dump(results, f)





def train_test(X, y, k=5, verbose=False, store=False):
    models = [svm.SVC(), svm.SVC(kernel='poly')] #, svm.SVC(kernel='linear')] #,svm.SVC(kernel='poly')]
    folds = kfold(len(X), k)

    results = []
    
    for model in models:
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

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc, prec, rec, f1 = metrics(y_pred, y_test)
            temp.append((acc, prec, rec, f1))
        
        avg_acc = sum(x[0] for x in results)/k
        avg_prec = sum(x[1] for x in results)/k
        avg_rec = sum(x[2] for x in results)/k
        avg_f1 = sum(x[3] for x in results)/k

        if verbose:
            print(f"kernel {model.kernel}")
            print(f"average accuracy {avg_acc}")
            print(f"average precision {avg_prec}")
            print(f"average recall {avg_rec}")
            print(f"average f1 {avg_f1}")
            print()
        
        results.append((avg_acc, avg_prec, avg_rec, avg_f1))

    if store:
        ls = []
        for model, res in zip(models, results):
            ls.append({
                "kernel": model.kernel,
                "avg accuracy": res[0],
                "avg precision": res[1],
                "average recall": res[2],
                "average f1": res[3]
            })

        with open("results/svm_results_by_kernel.json", "w") as f:
            json.dump(ls, f)



if __name__ == "__main__":
    df = pd.read_csv("my_data/labels.csv")
    X, y = data_prep(lda=True)
    #grid_search(X, y, verbose=True)
    train_test(X, y, verbose=True, store=True)