#!/usr/bin/env python
# coding: utf-8

# #**Proyecto - Sistemas Recomendadores - IIC3633**
# 
# ## Implementación en Keras de Session-Based RNNs for Recommendation
# 
# ### Utilizacion de Dwell Time como feature implicita para mejor rendimiento

# In[ ]:


import os
import sys
import subprocess
import math
import pandas as pd
import numpy as np
import sklearn
import psutil
import humanize
import GPUtil as GPU
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python.client import device_lib


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import warnings

import keras
import keras.backend as K
from keras.utils import to_categorical
from keras.losses import cosine_proximity, categorical_crossentropy
from keras.models import Model, Sequential
from keras.initializers import glorot_uniform
from keras.layers.core import Permute, Reshape, RepeatVector
from keras.layers import Input, Dense, Dropout, CuDNNGRU, Embedding, concatenate, Lambda, multiply, merge, Flatten
from keras.callbacks import ModelCheckpoint


# In[ ]:





# In[ ]:


class SessionDataset:
    def __init__(self, data, sep='\t', session_key='SessionId', item_key='ItemId', time_key='Time', n_samples=-1, itemmap=None, time_sort=False):
        """
        Args:
            path: path of the csv file
            sep: separator for the csv
            session_key, item_key, time_key: name of the fields corresponding to the sessions, items, time
            n_samples: the number of samples to use. If -1, use the whole dataset.
            itemmap: mapping between item IDs and item indices
            time_sort: whether to sort the sessions by time or not
        """
        self.df = data
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort
        self.add_item_indices(itemmap=itemmap)
        self.df.sort_values([session_key, time_key], inplace=True)

        #Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        #clicks within a session are next to each other, where the clicks within a session are time-ordered.

        self.click_offsets = self.get_click_offsets()
        self.session_idx_arr = self.order_session_idx()
        
        
    def get_click_offsets(self):
        """
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()

        return offsets
    

    def order_session_idx(self):
        """ Order the session indices """
        if self.time_sort:
            # starting time for each sessions, sorted by session IDs
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            # order the session indices by session starting times
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())

        return session_idx_arr
    
    
    def add_item_indices(self, itemmap=None):
        """ 
        Add item index column named "item_idx" to the df
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # unique item ids
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            itemmap = pd.DataFrame({self.item_key:item_ids,
                                   'item_idx':item2idx[item_ids].values})
        
        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')
        
    
    @property    
    def items(self):
        return self.itemmap.ItemId.unique()
        

class SessionDataLoader:
    def __init__(self, dataset, batch_size=50):
        """
        A class for creating session-parallel mini-batches.
        Args:
             dataset (SessionDataset): the session dataset to generate the batches from
             batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.done_sessions_counter = 0
        
        
    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        # initializations
        df = self.dataset.df
        session_key='SessionId'
        item_key='ItemId'
        time_key='TimeStamp'
        self.n_items = df[item_key].nunique()+1
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = [] # indicator for the sessions to be terminated
        finished = False        

        while not finished:
            minlen = (end - start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]
            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                input = idx_input
                target = idx_target
                yield input, target, mask
                
            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            self.done_sessions_counter = len(mask)
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]


# In[ ]:


# Modelo
def create_model():   
    emb_size = 50
    hidden_units = 100
    size = emb_size

    inputs = Input(batch_shape=(batch_size, 1, n_items))
    gru, gru_states = CuDNNGRU(hidden_units, stateful=True, return_state=True)(inputs)# drop1) #
    drop2 = Dropout(0.25)(gru)
    predictions = Dense(n_items, activation='softmax')(drop2)
    model = Model(input=inputs, output=[predictions])
    # lr original es 0.0001
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=categorical_crossentropy, optimizer=opt)
    model.summary()

    filepath='./DwellTimeModel_checkpoint.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = []
    return model


# In[ ]:


def get_states(model):
    return [K.get_value(s) for s,_ in model.state_updates]

def set_states(model, states):
    for (d,_), s in zip(model.state_updates, states):
        K.set_value(d, s)


# In[ ]:


from sklearn.metrics import recall_score

def get_recall(model, train_generator_map, recall_k=20):


    test_dataset = SessionDataset(test_data, itemmap=train_generator_map)
    test_generator = SessionDataLoader(test_dataset, batch_size=batch_size)


    n = 0
    suma = 0
    suma_baseline = 0

    for feat, label, mask in test_generator:

        
        print(feat)
        
        input_oh = to_categorical(feat, num_classes=loader.n_items) 
        input_oh = np.expand_dims(input_oh, axis=1)

        target_oh = to_categorical(label, num_classes=loader.n_items)

        pred = model.predict(input_oh, batch_size=batch_size)

        if n%100 == 0:
            try:
                print("{}:{}".format(n, suma/n))
            except:
                pass

        for row_idx in range(feat.shape[0]):
            pred_row = pred[row_idx] 
            label_row = target_oh[row_idx]

            idx1 = pred_row.argsort()[-recall_k:][::-1]
            idx2 = label_row.argsort()[-1:][::-1]

            n += 1
            if idx2[0] in idx1:
                suma += 1

    print("Recall@{} epoch {}: {}".format(recall_k, epoch, suma/n))


# In[ ]:


def get_mrr(model, train_generator_map, mrr_k=20):

    test_dataset = SessionDataset(test_data, itemmap = train_generator_map)
    test_generator = SessionDataLoader(test_dataset, batch_size=batch_size)

    n = 0
    suma = 0
    suma_baseline = 0

    for feat, label, mask in test_generator:
        input_oh = to_categorical(feat, num_classes=loader.n_items) 
        input_oh = np.expand_dims(input_oh, axis=1)
        target_oh = to_categorical(label, num_classes=loader.n_items)

        pred = model.predict(input_oh, batch_size=batch_size)

        if n%100 == 0:
            try:
                print("{}:{}".format(n, suma/n))
            except:
                pass

        for row_idx in range(feat.shape[0]):
            pred_row = pred[row_idx] 
            label_row = target_oh[row_idx]

            idx1 = pred_row.argsort()[-mrr_k:][::-1]
            idx2 = label_row.argsort()[-1:][::-1]

            n += 1
            if idx2[0] in idx1:
                suma += 1/int((np.where(idx1 == idx2[0])[0]+1))        

    print("MRR@{} epoch {}: {}".format(mrr_k, epoch, suma/n))


# In[ ]:


def train_model(model, save_weights = False, path_to_weights = False):
    train_dataset = SessionDataset(train_data)

    model_to_train = model

    with tqdm(total=train_samples_qty) as pbar:
        for epoch in range(1, 10):
            if path_to_weights:
                loader = SessionDataLoader(train_dataset, batch_size=batch_size)
            for feat, target, mask in loader:

                input_oh = to_categorical(feat, num_classes=loader.n_items) 
                input_oh = np.expand_dims(input_oh, axis=1)

                target_oh = to_categorical(target, num_classes=loader.n_items)

                tr_loss = model_to_train.train_on_batch(input_oh, target_oh)

                real_mask = np.ones((batch_size, 1))
                for elt in mask:
                    real_mask[elt, :] = 0

                hidden_states = get_states(model_to_train)[0]

                hidden_states = np.multiply(real_mask, hidden_states)
                hidden_states = np.array(hidden_states, dtype=np.float32)
                model_to_train.layers[1].reset_states(hidden_states)

                pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(epoch, tr_loss))
                pbar.update(loader.done_sessions_counter)

            # get metrics for epoch
            get_recall(model_to_train, train_dataset.itemmap)
            get_mrr(model_to_train, train_dataset.itemmap)

            # save model
            if save_weights:
                model_to_train.save('./DwellTimeEpoch{}.h5'.format(epoch))


# In[ ]:


if __name__ == '__main__':
    PATH_TO_TRAIN = '../DwellTimeTheano/augmented.csv'
    PATH_TO_DEV = '../processedData/rsc15_train_valid.txt'
    PATH_TO_TEST = '../processedData/rsc15_test.txt'
    train_data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    dev_data = pd.read_csv(PATH_TO_DEV, sep='\t', dtype={'ItemId':np.int64})
    test_data = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})
    
    
    batch_size = 512 #como en el paper
    session_max_len = 100
    embeddingp=False

    n_items = len(train_data['ItemId'].unique())+1
    print("Items unicos training:", n_items)

    dev_n_items = len(dev_data['ItemId'].unique())+1
    print("Items unicos dev:", dev_n_items)

    test_n_items = len(test_data['ItemId'].unique())+1
    print("Items unicos testing:", test_n_items)

    train_samples_qty = len(train_data['SessionId'].unique()) # cantidad sesiones no augmentadas de train
    print("Sesiones training:", train_samples_qty)

    dev_samples_qty = len(dev_data['SessionId'].unique()) # cantidad sesiones no augmentadas de dev
    print("Sesiones validation:",dev_samples_qty)

    test_samples_qty = len(test_data['SessionId'].unique()) # cantidad sesiones no augmentadas de test
    print("Sesiones testing:", test_samples_qty)
    
    train_fraction = 1#256 # 1/fraction es la cantidad de sesiones mas recientes a considerar
    dev_fraction = 1#2

    train_offset_step=train_samples_qty//batch_size
    dev_offset_step=dev_samples_qty//batch_size
    test_offset_step=test_samples_qty//batch_size
    aux = [0]
    aux.extend(list(train_data['ItemId'].unique()))
    itemids = np.array(aux)
    itemidmap = pd.Series(data=np.arange(n_items), index=itemids) 
    
    model = create_model()
    
    train_model(model)

