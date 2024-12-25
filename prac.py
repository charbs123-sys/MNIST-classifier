import numpy as np
import pandas as pd
import struct as st
import math

def convert_to_binary(path_images, path_labels):
    filename = {'images' : path_images ,'labels' : path_labels}
    train_imagesfile = open(filename['images'],'rb')
    train_imagesfile.seek(0)
    magic = st.unpack('>4B',train_imagesfile.read(4))
    nImg = st.unpack('>I',train_imagesfile.read(4))[0]
    nR = st.unpack('>I',train_imagesfile.read(4))[0]
    nC = st.unpack('>I',train_imagesfile.read(4))[0] 
    nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
    images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC))
    
    # Open labels file
    train_labelsfile = open(filename['labels'], 'rb')
    train_labelsfile.seek(0)

    # Read label metadata
    magic = st.unpack('>4B', train_labelsfile.read(4))
    nLabels = st.unpack('>I', train_labelsfile.read(4))[0]  # Number of labels

    # Read label data
    labels_array = np.asarray(
        st.unpack('>' + 'B' * nLabels, train_labelsfile.read(nLabels))
    )
    return images_array, labels_array


def RELU(A):
    if A > 0:
        return A
    else:
        return 0


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()
