# Brendan McShane Final Project
Supernova Detection


## TODO
- (histogram for the metrics of the different models?)
- logistic regression C plot?
- Start Report
- reconstruct images from PCs
https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
- try normalizing data?
- random forest?






## Challenges
Initially I wanted to reproduce the results of the attention-based FCOS model in the Supernovae Detection with Fully Convolutional One-Stage Framework paper from 2021, but quickly realized that it was a little too cutting-edge and complicated to be implemented by a one man team in the time frame that I had.

Instead, I decided to try to follow a paper from 2015 by Buisson et al. called Machine learning classification of SDSS transient survey images. The issue with this paper is that the data is from sky surveys from 2005-2007 that have thusfar been very hard to gain access too.

some issues with cropping and image capturing

Issues with:
cropping
0012_nova.png
0046
0056
0055
0091
0096
0118
0127
0128
0129

0118 non nova

looks like a supernova
0086


The next issue I ran into was I ended up getting comlpex eigenvalues during the PCA/LDA process. I'm currently looking into what that means for the data and algorithms but I'm assuming nothing good. I'm also assuming I've made a misstep along the way somewhere.


Less data because the images are in grayscale.


negative eigenvalues in LDA solution messes with scree plot and proportions


## Introduction

## Related Works

## Datasets


## Methodology


### PCA
https://towardsdatascience.com/principal-component-analysis-from-scratch-in-numpy-61843da1f967


### LDA
https://towardsdatascience.com/linear-discriminant-analysis-in-python-76b8b17817c2
https://machinelearningmastery.com/linear-discriminant-analysis-with-python/

### KNN
https://machinelearninginterview.com/topics/machine-learning/how-does-knn-algorithm-work-what-are-the-advantages-and-disadvantages-of-knn/

very simple, only parameter needing to be tuned is k (maybe distance method), no training,

sensitive to outliers, assumes equal importance of all features, high prediction complexity for high dimensions


### SVM
https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

good accuracy and fast prediction, less memory because we only care about support vectors during decision phase, kernels

long training time, works poorly with overlapping classes


### ANN
https://towardsdatascience.com/building-an-ann-with-tensorflow-ec9652a7ddd4

### Logistic Regression
https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

## Experiments

## Discussion


## Conclusions