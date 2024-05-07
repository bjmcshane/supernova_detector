import random


def metrics(y_pred, y_test):
    tp, tn, fp, fn = 0, 0, 0, 0
    n = len(y_pred)
    for pred, actual in zip(y_pred, y_test):
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 0 and actual == 0:
            tn +=1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1

    accuracy = (tp + tn)/n
    precision = tp/(tp+fp)
    recall = tp/(tp+tn)
    f1 = 2*(precision*recall)/(precision + recall)

    return accuracy, precision, recall, f1

def kfold(n, k):
    indices = list(range(n))
    random.shuffle(indices)
    folds = []

    fold_size = int(n/k)
    for i in range(k):
        folds.append(indices[int(i*(n/k)):int((i+1)*(n/k))])

    return folds
