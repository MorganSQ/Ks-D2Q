#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = "suqiang@kuaishou.com"

import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers,Model

BATCH_SIZE = 256
value_ranges = list(zip(range(0,16),[2000] * 16)) + [(16, 720)]
#print(value_ranges)

def make_parse_csv():
    def parse_csv(line):
        fields_defaults = [int()] * 17 + [float()] * 17
        fields = tf.io.decode_csv(line,fields_defaults)
        id_features = []
        for (index, _) in value_ranges:
            id_features.append(tf.clip_by_value(fields[index], clip_value_min = 0, clip_value_max = value_ranges[index][1] - 1))
        
        float_features = [fields[i] for i in range(17, 33)]
        features = tuple(id_features + float_features)
        labels = tf.clip_by_value(fields[33]/720.0,clip_value_min = 0, clip_value_max = 1.0)
        return features,labels
    return parse_csv

def make_dataset(filelist, repeat_times=1):
    dataset = tf.data.TextLineDataset(filelist).skip(1)
    dataset = dataset.map(make_parse_csv())
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(2 * BATCH_SIZE + 1)
    dataset = dataset.repeat(repeat_times)
    return dataset

train_files = ['./train_2.csv']
eval_files = ['./test.csv']

def D2Q():
    inputs=[]
    pxtr_sparse_concat = []
    for dim in list(zip(*value_ranges))[1]:
        if dim != 2000:
            continue
        x = layers.Input(shape=(1,),dtype=tf.int32)
        e = layers.Embedding(input_dim=dim, output_dim=32)(x)
        inputs.append(x)
        pxtr_sparse_concat.append(e)
    print("pxtr_sparse_concat=", tf.shape(pxtr_sparse_concat))
    
    duration_concat = []
    x = layers.Input(shape=(1,),dtype=tf.int32)
    e = layers.Embedding(input_dim=720, output_dim=32)(x)
    inputs.append(x)
    duration_concat.append(e)
    print("duration_concat=", tf.shape(duration_concat))
    
    pxtr_dense_concat = []
    for i in range(16):
        x = layers.Input(shape=(1,),dtype=tf.float32)
        inputs.append(x)
        pxtr_dense_concat.append(x)
    

    hidden_layer = []
    pxtr_sparse_emb = tf.concat(pxtr_sparse_concat,2)
    for dim in [512]:
        pxtr_sparse_emb = layers.Dense(dim,tf.nn.swish)(pxtr_sparse_emb)
    pxtr_sparse_emb = tf.reshape(pxtr_sparse_emb, [-1, 512])
    hidden_layer.append(pxtr_sparse_emb)
    
    duration_emb = tf.concat(duration_concat,2)
    for dim in [32]:
        duration_emb = layers.Dense(dim,tf.nn.swish)(duration_emb)
    duration_emb = tf.reshape(duration_emb, [-1, 32])
    hidden_layer.append(duration_emb)
    
    pxtr_dense_emb = tf.concat(pxtr_dense_concat,1)
    for dim in [32]:
        pxtr_dense_emb = layers.Dense(dim,tf.nn.swish)(pxtr_dense_emb)
    hidden_layer.append(pxtr_dense_emb)
    
    x = tf.concat(hidden_layer,1)
    
    for dim in [512,256,128,64]:
        x = layers.Dense(dim,tf.nn.swish)(x)
    out = layers.Dense(1, tf.sigmoid)(x)

    return Model(inputs = inputs, outputs = out)

def mse_loss(labels, output):
    loss = tf.losses.mean_squared_error(labels, output)
    return loss

def train():
    model = D2Q()
    model.compile(loss=mse_loss, optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1),)
    model.fit(make_dataset(train_files,1))
    model.save_weights('./model/d2q_model')
 
if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()
