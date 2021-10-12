# import the necessary packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import cv2

from math import sqrt, ceil
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram

import sys
import os
from os import path
import struct
from array import array


def predict_2d_for_plotting(x, y, func):
    X, Y = np.meshgrid(x, y)
    Xmat = np.column_stack([X.reshape(X.size), Y.reshape(Y.size)])
    Z = func(Xmat).reshape(x.size, y.size)
    return(X, Y, Z)

def plot_2d_prediction(x, y, func, xval, yval, figsize=(8, 8), midval=0):
    X, Y, Z = predict_2d_for_plotting(xval, yval, func)
    plt.figure(figsize=figsize)
    plt.contourf(X, Y, Z, cmap='bwr', levels=[Z.min(), midval, Z.max()], alpha=0.2)
    plt.scatter(np.array(x)[:, 0], np.array(x)[:, 1], c=y, cmap='bwr')
    plt.show()

def show_img(img, figwidth=10., cmap='gray', interpolation='bilinear'):
    plt.figure(figsize=(img.shape[1] * figwidth / img.shape[0], figwidth))
    print(img.shape)

    if len(img.shape) == 2:
        fig = plt.imshow(img, cmap=cmap, aspect='auto', interpolation=interpolation)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        fig = plt.imshow(img[:, :, 0], cmap=cmap, aspect='auto', interpolation=interpolation)
    elif len(img.shape) == 3 and img.shape[2] in [3, 4]:
        fig = plt.imshow(img, aspect='auto', interpolation=interpolation)
    else:
        raise ValueError('invalid image format')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()

def show_img_data(x, no=30, layout=(5, 6), figsize=(10, 10), interpolation='bilinear'):
    total = min(no, x.shape[0])
    size = layout[0] * layout[1]
    for i in range(ceil(total / size)):
        plt.figure(figsize=figsize)
        for j in range(size):
            if i * size + j >= total:
                break
            plt.subplot(layout[0], layout[1], j + 1)

            if len(x.shape) == 3:
                fig = plt.imshow(x[i * size + j, :, :], cmap='gray', interpolation=interpolation);
            elif len(x.shape) == 4 and x.shape[3] == 1:
                fig = plt.imshow(x[i * size + j, :, :, 0], cmap='gray', interpolation=interpolation);
            elif len(x.shape) == 4 and x.shape[3] in [3, 4]:
                fig = plt.imshow(x[i * size + j, :, :, :], interpolation=interpolation);
            else:
                raise ValueError('invalid image format')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        plt.show()
        
        
def plot_history(history, measure='accuracy', figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training history')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    if measure is not None and measure in history.history.keys() and \
        'val_' + measure in history.history.keys():
        plt.figure(figsize=figsize)
        plt.plot(history.history[measure])
        plt.plot(history.history['val_' + measure])
        plt.title('Training history')
        plt.ylabel(measure)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

def evaluate_classification_result(y, pred, classes=None, no_classes=10):
    if classes is not None:
        no_classes = len(classes)
    else:
        classes = [str(i) for i in range(no_classes)]

    if len(pred.shape) == 2:
        predC = np.argmax(pred, axis=1)
    elif len(pred.shape) == 1:
        predC = pred
    else:
        raise('pred has wrong format')
    cfTable = confusion_matrix(y, predC, labels=range(no_classes))
    
    print(cfTable)

    print('\n')
    
    TPRs = pd.Series(0., index=range(no_classes))
    
    for cl in range(no_classes):
        print('Class %s:'%(classes[cl]))
        others = list(set(list(range(cfTable.shape[0]))) - set([cl]))
        tp = cfTable[cl, cl]
        tn = np.sum(cfTable[others, :][:, others])
        fp = np.sum(cfTable[others, :][:, [cl]])
        fn = np.sum(cfTable[[cl], :][:, others])
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        prec = tp / (tp + fp)
        TPRs[cl] = tpr
        print('    Sensitivity (TPR): %7.3f%% (%d of %d)'%(100. * tpr, tp, tp + fn)) 
        print('    Specificity (TNR): %7.3f%% (%d of %d)'%(100. * tnr, tn, tn + fp)) 
        print('    Precision:         %7.3f%% (%d of %d)'%(100. * prec, tp, tp + fp))
        print('    Neg. pred. value:  %7.3f%% (%d of %d)'%(100. * tn / (tn + fn) , tn, tn + fn))
    
    print('\nOverall accuracy:  %7.3f%% (%d of %d)'%(np.sum(np.diagonal(cfTable)) * 100. / len(y),
          np.sum(np.diagonal(cfTable)), len(y)))
    print('Balanced accuracy: %7.3f%%'%(np.mean(TPRs) * 100.))
        
    return(cfTable)

## auxiliary function for evaluating regression results
def evaluate_regression_result(y, y_pred):
    print(f"Mean squared error (MSE): {mean_squared_error(y, y_pred).round(2)}")
    print(f"Root mean squared error (RMSE): {np.sqrt(mean_squared_error(y, y_pred)).round(2)}")
    print(f"Mean absolute error (MAE): {mean_absolute_error(y, y_pred).round(2)}")
    print(f"Coefficient of determination (R2): {r2_score(y, y_pred).round(2)}")

    cor = pearsonr(y, y_pred)
    print(f"Correlation coefficient (Pearson): {cor[0].round(2)} (p = {cor[1]})")

    
## auxiliary function for plotting agglomerative clustering results
## source: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html (+ modifications)
def plot_dendrogram(model, figsize=(8, 6), title='Hierarchical Clustering', **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=figsize)
    plt.title(title)
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Number of samples in node (or index of sample if no parenthesis)")
    plt.show()


## source: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py (+ modifications)
def read_MNIST(dataset="train", path = "./"):
    if dataset is "train":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "test":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'test' or 'train'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()
    img = 1. - np.array(img).reshape(len(lbl), rows, cols, 1) / 255.
    lbl = np.array(lbl).astype('int32')
    return lbl, img

def create_data_from_testimage(file, thicken=0):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # check for image format
    if img.shape[0] < img.shape[1]:
        raise Exception('Image is not in upright format, please rotate image!')

    # resize image
    dim = (2000, round(img.shape[0] * (2000 / img.shape[1])))
    img = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_LINEAR)

    # threshold image
    tmIm = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 501, 50)

    # isolate circles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clIm = cv2.morphologyEx(tmIm, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (70, 70))
    opIm = cv2.morphologyEx(clIm, cv2.MORPH_OPEN, kernel)

    # setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = False
    params.filterByInertia = False

    # create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # detect blobs
    keypoints = detector.detect(~opIm)

    # prepare warping
    pts1 = np.full((4, 2), np.nan, dtype='float32')
    pts2 = np.float32([[100, 100], [1900, 100], [100, 2700], [1900, 2700]])

    offset = 250
    for kp in keypoints:
        if (kp.pt[0] < offset) and (kp.pt[1] < offset):
            pts1[0] = kp.pt
        elif (kp.pt[0] > img.shape[1] - offset) and (kp.pt[1] < offset):
            pts1[1] = kp.pt
        elif (kp.pt[0] < offset) and (kp.pt[1] > img.shape[0] - offset):
            pts1[2] = kp.pt
        elif (kp.pt[0] > img.shape[1] - offset) and (kp.pt[1] > img.shape[0] - offset):
            pts1[3] = kp.pt

    if np.isnan(pts1).any():
        raise Exception('Could not find all markers.')

    # perform warping
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_n = cv2.warpPerspective(img, M, (2000, 2800))

    # read characters
    x = np.zeros(shape=(100, 28, 28, 1), dtype=np.float64)
    y = np.zeros(shape=(100,), dtype=np.int32)

    if thicken > 0:
        thicken = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thicken, thicken))
    else:
        thicken = None

    for i in range(10):
        for j in range(10):
            x1, y1 = 220 + i * 160, 440 + j * 200
            x2, y2 = x1 + 120, y1 + 120
            subimg = img_n[y1:y2, x1:x2]
            subthr = cv2.adaptiveThreshold(subimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 10)

            if thicken is not None:
                subthr = ~cv2.morphologyEx(~subthr, cv2.MORPH_DILATE, thicken)

            subscl = cv2.resize(subthr, (28, 28))
            x[j * 10 + i, :, :, 0] = subscl / 255.
            y[j * 10 + i] = j

    return(x, y, img_n)
