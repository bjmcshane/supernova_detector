
import numpy as np
from PIL import Image
import math
import statistics as stat
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_validate
import json
from pca import *
from lda import *


def data_prep(path="my_data/labels.csv"):
    df = pd.read_csv(path)
    y = df['label']
    matrix = np.zeros((len(df), 51*51))
    matrix[:] = np.NaN


    # in this loop I take all the grayscale pixel values for each image, flatten them into an array, and store them
    # in a (# of images) X (# of pixels) matrix
    for i in range(len(df)):
        arr = np.array(Image.open(df.iloc[i]["path"]), dtype="float")
        matrix[i] = arr.flatten()


    pca_A = PCA(matrix)
    #lda_A = LDA(matrix)

    X = pca_A

    return X, y



def kfold(n, k):
    indices = range(n)
    shuffled = np.random.shuffle(indices)
    folds = []

    fold_size = int(n/k)
    for i in range(k):
        folds.append(shuffled[i*(n/k):(i+1)*(n/k)])

    return folds






def cv(X, y, verbose=False, store=False):
    

    
    results = []
    for k, scores in enumerate(cv_scores):
        temp = {
            'avg_train_time': stat.mean(scores['fit_time']),
            'avg_test_time': stat.mean(scores['score_time']),
            'avg_test_acc': stat.mean(scores['test_acc']),
            'avg_test_prec': stat.mean(scores['test_prec_macro']),
            'avg_test_rec': stat.mean(scores['test_rec_macro']),
        }

        results.append(temp)

    if verbose:
        for i, res in enumerate(results):
            print(f"k: {i+1}")
            print(f"avg fit time: {res['avg_train_time']}")
            print(f"avg score time: {res['avg_test_time']}")
            print(f"avg test acc: {res['avg_test_acc']}")
            print(f"avg test precision: {stat.mean(scores['test_prec_macro'])}")
            print(f"avg test recall: {stat.mean(scores['test_rec_macro'])}")
            print()

    if store:
        with open("results/knn_results_by_k.json", "w") as f:
            json.dump(results, f)
            
    return results





if __name__ == "__main__":
    param_grid = {
        "M" : [5,10,15],
        'k' : [5,7,9]
    }


    #X, y = data_prep()
    #cv(X, y, store=True)
    print(kfold(10, 3))