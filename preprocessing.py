from concurrent.futures import process
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import sys
import cv2
import math
import random
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore') # ignore warnings about complex numbers in PCA/LDA

images = "dataverse_files/PSP/images"
annotations = "dataverse_files/PSP/Annotations"
sets = "dataverse_files/PSP/ImageSets"
labels = "dataverse_files/PSP/labels/"

# a list of all the string paths to all PSP images
images = [f for f in listdir(images)]
images.sort()

# list of all .xml file annotations for the above images. They look something like the below comment
annotations = [f for f in listdir(annotations)]
annotations.sort()
'''
<annotation>
	<folder>images</folder>
	<filename>0001.png</filename>
	<path>E:\Kyle\dataset\data716\images\0001.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>344</width>
		<height>296</height>
		<depth>1</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>nova</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>71</xmin>
			<ymin>212</ymin>
			<xmax>81</xmax>
			<ymax>222</ymax>
		</bndbox>
	</object>
</annotation>
'''


# reads in images from the Popular Supernova Project (PSP) image database, crops out a supernova 
# # and a non-supernova window from each, and then stores them in the my_data/ folder along with a
# csv file that contains the individual image paths and their corresponding labels
def process_PSP(display=False):
    df = pd.DataFrame(columns=["path","label"],index=range(0,len(images)*2))
    window_dimension = 51
    i=0
    for image, annotation in zip(images, annotations):
        im = Image.open("dataverse_files/PSP/images/"+image).convert('L')
        im_width, im_height = im.width, im.height


        with open("dataverse_files/PSP/Annotations/"+annotation, 'r') as f:
            label = f.read()
        

        bs_data = BeautifulSoup(label, 'xml')
        xmin = int(bs_data.findChild('xmin').string)
        xmax = int(bs_data.findChild('xmax').string)
        ymin = int(bs_data.findChild('ymin').string)
        ymax = int(bs_data.findChild('ymax').string)


        # supernova window
        width = xmax-xmin
        height = ymax-ymin

        left_bumper = math.ceil((window_dimension-width)/2)
        right_bumper = math.floor((window_dimension-width)/2)
        above_bumper = math.ceil((window_dimension-height)/2)
        below_bumper = math.floor((window_dimension-height)/2)

        new_xmin = xmin - left_bumper
        new_xmax = xmax + right_bumper
        new_ymin = ymin - above_bumper
        new_ymax = ymax + below_bumper

        if new_xmin < 0 or new_ymin < 0 or new_xmax > im_width or new_ymax > im_height:
            continue


        nova = im.crop((new_xmin, new_ymin, new_xmax, new_ymax))
        nova_path = "my_data/" + image.split('.')[0]+"_nova."+image.split('.')[1]

        # non-supernova window
        if xmin > 52:
            nnew_xmin=0
            nnew_xmax=51
        else:
            nnew_xmin = im_width-52
            nnew_xmax = im_width-1

        if ymin > 52:
            nnew_ymin = 0
            nnew_ymax = 51
        else:
            nnew_ymin = im_height-52
            nnew_ymax = im_height-1
            

        non_nova = im.crop((nnew_xmin, nnew_ymin, nnew_xmax, nnew_ymax))
        non_nova_path = "my_data/" + image.split('.')[0]+"_non_nova."+image.split('.')[1]

        nova.save(nova_path)
        non_nova.save(non_nova_path)
        df.iloc[i] = pd.Series({"path": nova_path, "label": 1})
        df.iloc[i+1] = pd.Series({"path": non_nova_path, "label": 0})
        i+=2
        if image == "0002.png":
            draw = ImageDraw.Draw(im)
            draw.line(((xmin,ymin),(xmin,ymax)), fill=0)
            draw.line(((xmin,ymin),(xmax,ymin)), fill=0)
            draw.line(((xmax,ymax),(xmax,ymin)), fill=0)
            draw.line(((xmax,ymax),(xmin,ymax)), fill=0)

            draw.line(((new_xmin,new_ymin),(new_xmin,new_ymax)), fill=0)
            draw.line(((new_xmin,new_ymin),(new_xmax,new_ymin)), fill=0)
            draw.line(((new_xmax,new_ymax),(new_xmax,new_ymin)), fill=0)
            draw.line(((new_xmax,new_ymax),(new_xmin,new_ymax)), fill=0)

            draw.line(((nnew_xmin,nnew_ymin),(nnew_xmin,nnew_ymax)), fill=0)
            draw.line(((nnew_xmin,nnew_ymin),(nnew_xmax,nnew_ymin)), fill=0)
            draw.line(((nnew_xmax,nnew_ymax),(nnew_xmax,nnew_ymin)), fill=0)
            draw.line(((nnew_xmax,nnew_ymax),(nnew_xmin,nnew_ymax)), fill=0)
            im.show()
            nova.show()
            non_nova.show()
            display=False

    df = df.dropna(axis=0)
    df.to_csv("my_data/labels.csv")



# takes all of the grayscale images stored in my_data and flattens them into individual rows and stores them
# in the matrix X along with their corresponding labels in y
def flatten_images(path="my_data/labels.csv"):
    # The dataframe is just two columns, path to the input image and it's corresponding label (supernova or not a supernova)
    df = pd.read_csv(path)
    y = df['label']
    matrix = np.zeros((len(df), 51*51))
    matrix[:] = np.NaN

    # in this loop I take all the grayscale pixel values for each image, flatten them into an array, and store them
    # in a (# of images) X (# of pixels) matrix
    for i in range(len(df)):
        arr = np.array(Image.open(df.iloc[i]["path"]), dtype="float")
        flat = arr.flatten()
        matrix[i] = flat


    return matrix, y


# a simple scree plot function to visualize how much variance each principal component accounts for
# in decreasing order
def scree_plot(e_values, lda=False):
    temp = e_values[:8].real
    sum_eigenvalues = np.sum(temp)
    prop_var = [i/sum_eigenvalues for i in temp]
    cum_var = [np.sum(prop_var[:i+1]) for i in range(len(prop_var))]


    x_labels = ["PC{}".format(i+1) for i in range(len(prop_var))]

    plt.plot(x_labels, prop_var, color='skyblue', linewidth=2, label='Proportion of variance')
    plt.plot(x_labels, cum_var, color='orange', linewidth=2, label='Cumulative variance')
    plt.legend()
    if lda:
        plt.title('LDA Scree Plot')
    else:
        plt.title('PCA Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Proportion of Variance')
    plt.show()



# performs PCA on the (# of images)x(# of pixels per image) matrix in order to extract important features
# and reduce dimensionality. M here is the number if principal components we're interested in using for prediction
# and image reconstruction
def PCA(matrix, M=10, scree=False):
    # self explanatory, computing the covariance matrix for the my input matrix
    cov = np.cov(matrix, rowvar=False)

    # eigendecomposition, this is where the complex numbers start coming into play
    e_values, e_vectors = np.linalg.eig(cov)

    # sort eigenvalues and eigenvecotors
    idx = e_values.argsort()[::-1]
    e_values = e_values[idx]
    e_vectors = e_vectors[:,idx]

    if scree:
        scree_plot(e_values)

    D = np.diag(e_values)
    P = e_vectors

    new_mat = np.dot(cov, P)

    new_eigenvals = e_values[:M]
    new_eigenvecs = e_vectors[:,:M]


    A = np.zeros((len(matrix), M))


    x_means = np.mean(matrix, axis=0)

    for i in range(len(matrix)):
        A[i] = np.dot(new_eigenvecs.transpose(), (matrix[i]-x_means).transpose())
    

    return A


def image_reconstruction(new_X, eigenvectors):
    return new_X @ eigenvectors.T

def image_reconstruction2(X, eigenvectors):
    return X @ eigenvectors @ eigenvectors.T

def reconstruction_error(X, new_X):
    return np.sqrt((X - new_X)**2/(X.shape[0]*X.shape[1]))


# performs LDA on the (# of images)x(# of pixels per image) matrix in order to extract important features
# and reduce dimensionality. M here is the number if components we're interested in using for prediction
def LDA(matrix, y, M=1, scree=False):
    rows, cols = matrix.shape
    S_w = np.zeros((cols, cols))
    S_b = np.zeros((cols, cols))

    x_means = np.mean(matrix, axis=0)

    for cl in [0, 1]:
        S_cl = np.zeros((cols, cols))

        idx = np.where(y==cl)
        temp = matrix[idx]

        temp_means = np.mean(temp, axis=0)

        for i in range(len(temp)):
            curr = temp[i]

            S_cl += (curr - temp_means).dot((curr - temp_means).T)

        S_w += S_cl

        n = len(temp)

        S_b += n*(temp_means - x_means).dot((temp_means - x_means).T)

        #S_w_inv = np.linalg.inv(S_w)
        S_w_inv = np.linalg.pinv(S_w)
        e_values, e_vectors = np.linalg.eig(S_w_inv.dot(S_b))


        # sort eigenvalues and eigenvecotors
        idx = e_values.argsort()[::-1]
        e_values = e_values[idx]
        e_vectors = e_vectors[:,idx]

        #print(f"sorted eigenvalues {e_values}")

        if scree:
            scree_plot(e_values, lda=True)


        D = np.diag(e_values)
        P = e_vectors

        new_mat = np.dot(1, P)

        sorted_eigenvals = e_values[:M]
        sorted_eigenvecs = e_vectors[:,:M]


        A = np.zeros((len(matrix), M))


        x_means = np.mean(matrix, axis=0)

        for i in range(len(matrix)):
            A[i] = np.dot(sorted_eigenvecs.transpose(), (matrix[i]-x_means).transpose())

        
        return A

# normalizes a given matrix X column-wise
def normal(X):
    return normalize(X, axis=0, norm='max')

# this method is used in almost every other file in this directory. It takes specifications for whether
# or not we want to include the LDA component and if we want to normalize the matrix resulting from our
# feature extraction process
def data_prep(path="my_data/labels.csv", lda=False, norm=False, pscree=False, lscree=False):
    X, y = flatten_images(path)
    
    if lda:
        X = np.concatenate((PCA(X, scree=pscree),LDA(X, y, scree=lscree)), axis=1)
    else:
        X = PCA(X, scree=pscree)

    if norm:
        X = normalize(X)

    return X, y


if __name__ == '__main__':
    #data_prep(lda=True)
    im = Image.open("dataverse_files/PSP/images/0002.png")
    #im.show()
    df = pd.read_csv("my_data/labels.csv")
    print(len(df))