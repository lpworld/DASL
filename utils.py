import os, sys, time, random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

batch_size=32

def dictionary(terms):
    term2idx = {}
    idx2term = {}
    for i in range(len(terms)):
        term2idx[terms[i]] = i
        idx2term[i] = terms[i]
    return term2idx, idx2term

class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1
        u, hist, hist_cross, i, y = [], [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            hist_cross.append(t[2])
            i.append(t[3])
            y.append(t[4])
        return (u, hist, hist_cross, i, y)

def compute_auc(sess, model, testset1, testset2):
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    arr_1, arr_2 = [], []
    for uij_1, uij_2 in zip(DataInput(testset1, batch_size), DataInput(testset2, batch_size)):
        a, b = model.test(sess, uij_1, uij_2)
        score, label, user = a
        #print(score)
        for index in range(len(score)):
            if label[index] > 0:
                arr_1.append([0, 1, score[index]])
            elif label[index] == 0:
                arr_1.append([1, 0, score[index]])
        score, label, user = b
        for index in range(len(score)):
            if label[index] > 0:
                arr_2.append([0, 1, score[index]])
            elif label[index] == 0:
                arr_2.append([1, 0, score[index]])
    arr_1 = sorted(arr_1, key=lambda d:d[2])
    arr_2 = sorted(arr_2, key=lambda d:d[2])
    auc_1 = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr_1:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc_1 += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr_1) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        auc_1 = -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        auc_1 = (1.0 - auc_1 / (2.0 * tp2 * fp2))
    else:
        auc_1 = None
        
    auc_2 = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr_2:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc_2 += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr_2) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        auc_2 = -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        auc_2 = (1.0 - auc_2 / (2.0 * tp2 * fp2))
    else:
        auc_2 = -0.5
    
    return auc_1, auc_2

def compute_hr(sess, model, testset1, testset2):
    hit_1, arr_1, hit_2, arr_2 = [], [], [], []
    userid1 = list(set([x[0] for x in testset1]))
    userid2 = list(set([x[0] for x in testset2]))
    for uij_1, uij_2 in zip(DataInput(testset1, batch_size), DataInput(testset2, batch_size)):
        a, b = model.test(sess, uij_1, uij_2)
        score, label, user = a
        for index in range(len(score)):
            if score[index] > 0.5:
                arr_1.append([label[index], 1, user[index]])
            else:
                arr_1.append([label[index], 0, user[index]])
        score, label, user = b
        for index in range(len(score)):
            if score[index] > 0.5:
                arr_2.append([label[index], 1, user[index]])
            else:
                arr_2.append([label[index], 0, user[index]])
    for user in userid1:
        arr_user = [x for x in arr_1 if x[2]==user and x[1]==1]
        if len(arr_user) > 0:
            hit_1.append(sum([x[0] for x in arr_user])/len(arr_user))
    for user in userid2:
        arr_user = [x for x in arr_2 if x[2]==user and x[1]==1]
        if len(arr_user) > 0:
            hit_2.append(sum([x[0] for x in arr_user])/len(arr_user))
    return np.mean(hit_1), np.mean(hit_2)