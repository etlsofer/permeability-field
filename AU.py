import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
# %matplotlib inline
import glob
import numpy as np
import imageio
import matplotlib.pyplot as plt


def convert_str_to_float(list_str):
    res = []
    for str in list_str:
        if str == "":
            continue
        res.append(float(str))
    return res


def min_max_normal():
    return


def read_mat(file_name):
    res = []
    list_of_lists = []
    with open(file_name) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            # in alternative, if you need to use the file content as numbers
            # inner_list = [int(elt.strip()) for elt in line.split(',')]
            list_of_lists.append(inner_list)

    for item in list_of_lists:
        res.append(convert_str_to_float(item))
    return np.asarray(res)


def prepare_train_data(test_dir):
    classes = []
    labels = []
    for file in glob.glob(test_dir + "\*.txt"):
        classes.append(read_mat(file))
    for file in glob.glob(test_dir + "\*.dat"):
        labels.append(read_mat(file))
    # convert to ndarray
    # classes, labels = np.asarray(classes, dtype=None, order=None),np.asarray(labels, dtype=None, order=None)
    return classes, labels

def list_of_res(c_mats,l_mats):
    res = []
    for mat1,mat2 in zip(c_mats,l_mats):
        temp = np.linalg.pinv(mat1)
        res.append(np.matmul(temp, mat2))
    return res


def main():
    # prepare data
    train_classes, train_labels = prepare_train_data(
        r"C:\Users\leah2\OneDrive\שולחן העבודה\hw\Lab Projects\muchine learnning"
        r"\Var 1\train")
    test_classes, test_labels = prepare_train_data(
        r"C:\Users\leah2\OneDrive\שולחן העבודה\hw\Lab Projects\muchine learnning"
        r"\Var 1\test")
    temp = list_of_res(train_classes,train_labels)
    print(len(temp), temp[0].shape)

    for i in range(len(temp)):
        print(train_labels[i].shape,train_classes[i].shape)
        #print(np.linalg.norm(temp[i]-temp[i+1]))
        #print(np.linalg.det(temp[i]-temp[i+1]))
        #s1,v1,d1 = np.linalg.svd(temp[i])
        #s2, v2, d2 = np.linalg.svd(temp[i+1])
        #print(s1*v1*np.transpose(d1)-temp[i])
        #fig, ax = plt.subplots(1, 2, figsize=(12, 10))
        #ax[0].imshow(s1*v1*d1)
        #ax[1].imshow(temp[i])
        #plt.show()
        #print(np.diagonal(v1))
        #print(np.matmul(s1,np.transpose(s1)))
        #print(np.linalg.norm(v1))
        #print(np.linalg.norm(d1))

    '''    
    val_size = 15
    train_size = len(train_classes) - val_size
    train_ds, val_ds = random_split(train_classes, [train_size, val_size])
    print(train_classes[0].shape)
    print(type(train_ds[0]), type(val_ds[0]), type(train_classes[0]))
    #c




    
    for image1 in train_ds:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(image1)
        #ax[2].imshow(image2)
        plt.show()
    '''



if __name__ == '__main__':
    main()



