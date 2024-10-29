import scipy.io as sio
import h5py
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
from pycocotools.coco import COCO
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
import h5py
import os


def load_deep_features(data_name):
    valid_data = True
    np.random.seed(1)
    if data_name == 'xmedia':
        path = 'datasets/XMedia/XMediaFeatures.mat'
        all_data = sio.loadmat(path)
        I_te_CNN = all_data['I_te_CNN'].astype('float32')   # Features of test set for image data, CNN feature
        I_tr_CNN = all_data['I_tr_CNN'].astype('float32')   # Features of training set for image data, CNN feature
        T_te_BOW = all_data['T_te_BOW'].astype('float32')   # Features of test set for text data, BOW feature
        T_tr_BOW = all_data['T_tr_BOW'].astype('float32')   # Features of training set for text data, BOW feature
        V_te_CNN = all_data['V_te_CNN'].astype('float32')   # Features of test set for video(frame) data, CNN feature
        V_tr_CNN = all_data['V_tr_CNN'].astype('float32')   # Features of training set for video(frame) data, CNN feature
        A_te = all_data['A_te'].astype('float32')           # Features of test set for audio data, MFCC feature
        A_tr = all_data['A_tr'].astype('float32')           # Features of training set for audio data, MFCC feature
        d3_te = all_data['d3_te'].astype('float32')         # Features of test set for 3D data, LightField feature
        d3_tr = all_data['d3_tr'].astype('float32')         # Features of training set for 3D data, LightField feature

        teImgCat = all_data['teImgCat'].reshape([-1,1]).astype('int64') # category label of test set for image data
        trImgCat = all_data['trImgCat'].reshape([-1,1]).astype('int64') # category label of training set for image data
        teVidCat = all_data['teVidCat'].reshape([-1,1]).astype('int64') # category label of test set for video(frame) data
        trVidCat = all_data['trVidCat'].reshape([-1,1]).astype('int64') # category label of training set for video(frame) data
        teTxtCat = all_data['teTxtCat'].reshape([-1,1]).astype('int64') # category label of test set for text data
        trTxtCat = all_data['trTxtCat'].reshape([-1,1]).astype('int64') # category label of training set for text data
        te3dCat = all_data['te3dCat'].reshape([-1,1]).astype('int64')   # category label of test set for 3D data
        tr3dCat = all_data['tr3dCat'].reshape([-1,1]).astype('int64')   # category label of training set for 3D data
        teAudCat = all_data['teAudCat'].reshape([-1,1]).astype('int64') # category label of test set for audio data
        trAudCat = all_data['trAudCat'].reshape([-1,1]).astype('int64') # category label of training set for audio data


        train_data = [I_tr_CNN, T_tr_BOW, A_tr, d3_tr, V_tr_CNN]
        test_data = [I_te_CNN[0: 500], T_te_BOW[0: 500], A_te[0: 100], d3_te[0: 50], V_te_CNN[0: 87]]
        valid_data = [I_te_CNN[500::], T_te_BOW[500::], A_te[100::], d3_te[50::], V_te_CNN[87::]]
        train_labels = [trImgCat, trTxtCat, trAudCat, tr3dCat, trVidCat]
        test_labels = [teImgCat[0: 500], teTxtCat[0: 500], teAudCat[0: 100], te3dCat[0: 50], teVidCat[0: 87]]
        valid_labels = [teImgCat[500::], teTxtCat[500::], teAudCat[100::], te3dCat[50::], teVidCat[87::]]

    elif data_name == 'xmedianet': # label 1*n
        valid_len = 4000
        path = 'datasets/XMediaNet/XMediaNet5View_Doc2Vec.mat'
        all_data = sio.loadmat(path)
        all_train_data = all_data['train'][0]
        all_train_labels = all_data['train_labels'][0]
        all_valid_data = all_data['valid'][0]
        all_valid_labels = all_data['valid_labels'][0]
        all_test_data = all_data['test'][0]
        all_test_labels = all_data['test_labels'][0]

        train_data, valid_data, test_data, train_labels, valid_labels, test_labels = [],[],[],[],[],[]
        for i in range(5):
            train_data.append(all_train_data[i].astype('float32'))
            train_labels.append(all_train_labels[i].reshape([-1,1]))
            valid_data.append(all_valid_data[i].astype('float32'))
            valid_labels.append(all_valid_labels[i].reshape([-1,1]))
            test_data.append(all_test_data[i].astype('float32'))
            test_labels.append(all_test_labels[i].reshape([-1,1]))
    elif data_name == 'wiki':
        valid_len = 231
        path = 'datasets/Wiki/wiki.mat'
        data = sio.loadmat(path)
        img_train = data['train_imgs_deep']
        text_train = data['train_texts_doc']
        label_train = data['train_imgs_labels'].reshape([-1,1])

        img_test = data['test_imgs_deep']
        text_test = data['test_texts_doc']
        label_test = data['test_imgs_labels'].reshape([-1,1])

        img_val = img_test[0:valid_len]
        text_val = text_test[0:valid_len]
        label_val = label_test[0:valid_len]

        img_test = img_test[valid_len:]
        text_test = text_test[valid_len:]
        label_test = label_test[valid_len:]

        train_data = [img_train, text_train]
        test_data = [img_test, text_test]
        valid_data = [img_val, text_val]
        train_labels = [label_train, label_train]
        test_labels = [label_test, label_test]
        valid_labels =  [label_val, label_val]
    elif data_name == 'nus':
        path = 'datasets/NUSWIDE/nus_wide_deep_doc2vec-corr-ae.h5py'
        with h5py.File(path, 'r') as file:
            img_train = file['train_imgs_deep'][:]
            text_train = file['train_texts'][:]
            label_train = file['train_imgs_labels'][:]

            img_val = file['valid_imgs_deep'][:]
            text_val = file['valid_texts'][:]
            label_val = file['valid_imgs_labels'][:]

            img_test = file['test_imgs_deep'][:]
            text_test = file['test_texts'][:]
            label_test = file['test_imgs_labels'][:]

            train_data = [img_train, text_train]
            test_data = [img_test, text_test]
            valid_data = [img_val, text_val]
            train_labels = [label_train, label_train]
            test_labels = [label_test, label_test]
            valid_labels =  [label_val, label_val]
    elif data_name == 'INRIA-Websearch':
        path = 'datasets/INRIA-Websearch/INRIA-Websearch.mat'
        data = sio.loadmat(path)
        img_train = data['tr_img'].astype('float32')
        text_train = data['tr_txt'].astype('float32')
        label_train = data['tr_img_lab'].reshape([-1,1]).astype('int16')

        img_val = data['val_img'].astype('float32')
        text_val = data['val_txt'].astype('float32')
        label_val = data['val_img_lab'].reshape([-1,1]).astype('int16')

        img_test = data['te_img'].astype('float32')
        text_test = data['te_txt'].astype('float32')
        label_test = data['te_img_lab'].reshape([-1,1]).astype('int16') 

        train_data = [img_train, text_train]
        test_data = [img_test, text_test]
        valid_data = [img_val, text_val]
        train_labels = [label_train, label_train]
        test_labels = [label_test, label_test]
        valid_labels =  [label_val, label_val]
    if valid_data:
        return train_data, train_labels, valid_data, valid_labels, test_data, test_labels
    else:
        return train_data, train_labels, test_data, test_labels
