# import the necessary packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import cv2

from math import sqrt, ceil, log10, floor
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

def plot_2d_prediction(x, y, func, xval, yval, label=None, figsize=(8, 8), midval=0):
    X, Y, Z = predict_2d_for_plotting(xval, yval, func)

    plt.figure(figsize=figsize)
    if label and len(label):
        plt.title(label)
    levels = [Z.min(), midval, Z.max()]
    if levels[0] >= levels[1]:
        levels[0] = levels[1] - 1
    if levels[2] <= levels[1]:
        levels[2] = levels[1] + 1
    plt.contourf(X, Y, Z, cmap='bwr', levels=levels, alpha=0.2)
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
        
## auxiliary function for creating meaningful tick positions for history plot
def create_ticks(n, base=5):
    if n == 1:
        return (np.arange(1, dtype='int') + 1)
    steps = ceil(n / base)
    magnitude = floor(log10(steps))
    choices = np.array([1, 2, 5, 10])
    step_c = choices * 10.**magnitude
    dist = np.abs(step_c - steps)
    sel = np.argmin(dist)
    steps = step_c[sel]
    ticks = np.arange(steps, n, steps, dtype='int')
    if ticks[0] > 1:
        ticks = np.concatenate(([1], ticks))
    if ticks[-1] < n:
        ticks = np.concatenate((ticks, [n]))
    return(ticks)

def plot_history(history, measure='accuracy', figsize=(8, 6)):
    epochs = [int(i + 1) for i in history.epoch]
    n = epochs[-1]
    ticks = create_ticks(n)
    plt.figure(figsize=figsize)
    if n == 1:
        plt.plot(epochs, history.history['loss'], 'o')
        plt.plot(epochs, history.history['val_loss'], 'o')
    else:
        plt.plot(epochs, history.history['loss'])
        plt.plot(epochs, history.history['val_loss'])
    plt.title('Training history')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(ticks)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    if measure is not None and measure in history.history.keys() and \
        'val_' + measure in history.history.keys():
        plt.figure(figsize=figsize)
        if n == 1:
            plt.plot(epochs, history.history[measure], 'o')
            plt.plot(epochs, history.history['val_' + measure], 'o')
        else:
            plt.plot(epochs, history.history[measure])
            plt.plot(epochs, history.history['val_' + measure])
        plt.title('Training history')
        plt.ylabel(measure)
        plt.xlabel('epoch')
        plt.xticks(ticks)
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

def evaluate_classification_result(y, pred, classes=None, no_classes=2,
                                   cutoff=0.5, hide_cm=False, return_cm=False):
    if classes is not None:
        no_classes = len(classes)
    else:
        classes = [str(i) for i in range(no_classes)]

    if len(pred.shape) == 2:
        if pred.shape[1] == no_classes:
            predC = np.argmax(pred, axis=1)
        elif pred.shape[1] == 1 and no_classes == 2:
            predC = (pred[:, [0]] >= cutoff)
        else:
            raise ValueError('pred has wrong format')
    elif len(pred.shape) == 1:
        predC = pred
    else:
        raise ValueError('pred has wrong format')
    cfTable = confusion_matrix(y, predC, labels=range(no_classes))

    if not hide_cm:
        print('Confusion matrix (rows -> true, columns -> predicted):\n')
        cfTable_pd = pd.DataFrame(cfTable, index=classes, columns=classes)
        print(cfTable_pd)
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
    
    if return_cm:
        return(cfTable)
    else:
        return

## auxiliary function for evaluating regression results
def evaluate_regression_result(y, y_pred, target_names=None):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    if len(y.shape) > 2 or len(y.shape) < 1:
        raise ValueError('y has the wrong format')
    if len(y_pred.shape) > 2 or len(y_pred.shape) < 1:
        raise ValueError('y_pred has the wrong format')
    if len(y.shape)  == 1:
        n = y.shape[0]
        y = y.reshape((n, 1))
    if len(y_pred.shape)  == 1:
        n = y_pred.shape[0]
        y_pred = y_pred.reshape((n, 1))
    if y.shape[0] != y_pred.shape[0]:
        raise ValueError('y and y_pred have different numbers of samples')
    if y.shape[1] != y_pred.shape[1]:
        raise ValueError('y and y_pred have different numbers of features')

    if target_names is None:
        target_names = [str(i) for i in range(0, y.shape[1])]

    if len(target_names) != y.shape[1]:
        raise ValueError('length of target_names is not compatible with number of columns in y and y_pred')

    for i, feat in enumerate(target_names):
        print('Target feature %s:'%(feat))
        print('    Mean squared error (MSE):          %7.3f'%(mean_squared_error(y[:, i], y_pred[:, i])))
        print('    Root mean squared error (RMSE):    %7.3f'%(np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))))
        print('    Mean absolute error (MAE):         %7.3f'%(mean_absolute_error(y[:, i], y_pred[:, i])))
        print('    Coefficient of determination (R2): %7.3f'%(r2_score(y[:, i], y_pred[:, i])))

        cor = pearsonr(y[:, i], y_pred[:, i])
        print('    Correlation coefficient (Pearson): %7.3f (p = %.2e)'%(cor[0], cor[1]))
        if i < len(target_names) - 1:
            print('')

    
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

def create_data_from_testimage(file, thicken=0, rotate=0):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError('Could not read from', file)

    if rotate == 0:
        pass
    elif rotate == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotate == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError('Invalid rotation angle', rotate)

    # check for image format
    if img.shape[0] < img.shape[1]:
        raise ValueError('Image is not in upright format, please rotate image!')

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
