#!/usr/bin/env python
# coding: utf-8

# VERSION PARA DEBUGGING, IGNORAR


import os
import pandas as pd
import numpy as np
import psutil
import humanize
import GPUtil as GPU
from tensorflow.python.client import device_lib


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import keras.backend as K
from keras.utils import to_categorical
from keras.losses import cosine_proximity
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Dropout, CuDNNGRU, GRU, Embedding, Flatten, Input
from keras.callbacks import ModelCheckpoint

GPUs = GPU.getGPUs()
gpu = GPUs[0]

def print_gpu_info():
  process = psutil.Process(os.getpid())
  print("Gen RAM Free: " + humanize.naturalsize(
          psutil.virtual_memory().available), " I Proc size: "  +
          humanize.naturalsize(process.memory_info().rss))
  print("GPU RAM Free {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total          {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, 
                           gpu.memoryTotal))
  
print_gpu_info()

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [(x.name, x.DESCRIPTOR, x.DEVICE_TYPE_FIELD_NUMBER, x.NAME_FIELD_NUMBER, x.PHYSICAL_DEVICE_DESC_FIELD_NUMBER) for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()

# Cargamos dataframes preprocesados de RSC15
PATH_TO_TRAIN = '../processedData/rsc15_train_tr.txt'
PATH_TO_DEV = '../processedData/rsc15_train_valid.txt'
PATH_TO_TEST = '../processedData/rsc15_test.txt'

train_data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
dev_data = pd.read_csv(PATH_TO_DEV, sep='\t', dtype={'ItemId':np.int64})
test_data = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})


# In[4]:


batch_size = 512 # como en el paper
session_max_len = 19

n_items = len(train_data['ItemId'].unique())+1
print("Items unicos:", n_items)

train_samples_qty = len(train_data['SessionId'].unique()) # cantidad sesiones no augmentadas de train
print("Sesiones training:", train_samples_qty)

dev_samples_qty = len(dev_data['SessionId'].unique()) # cantidad sesiones no augmentadas de dev
print("Sesiones validation:",dev_samples_qty)

test_samples_qty = len(test_data['SessionId'].unique()) # cantidad sesiones no augmentadas de test
print("Sesiones testing:", test_samples_qty)


# In[5]:


def process_pd(data):
    item_key = 'ItemId'
    session_key = 'SessionId'
    time_key = 'Time'
    
    itemids = data[item_key].unique()
    n_items = len(itemids)
    
    itemidmap = pd.Series(data=np.arange(n_items), index=itemids) # Mapeo desde los 37.5k a (0, 37.5k) id
    data = pd.merge(data, pd.DataFrame({item_key:itemids, 'ItemIdx':itemidmap[itemids].values}), on=item_key, how='inner') # agrego esa columna
    data.sort_values([session_key, time_key], inplace=True) # ordenamos por sesion
    
    # arreglo con offset acumulativo de inicio de cada sesion
    offset_sessions = np.zeros(data[session_key].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = data.groupby(session_key).size().cumsum()
    
    return data, itemidmap, offset_sessions # revitemidmap, 

training_data, training_itemidmap, training_offset_sessions = process_pd(train_data)


# In[66]:
# MODEL SHOULD BE A PARAMETER FOR GRU RESETTING
def batch_generator(data, itemidmap, offset_sessions, batch_size=128, session_max_len=19, fraction=1, offset=0):
    # Eventualmente intentar volver a implementar fracciones recientes del dataset
    item_key = 'ItemId'
    mapped_item_key = 'ItemIdx'
    session_key = 'SessionId'
    time_key = 'Time'
    n_items = len(data['ItemId'].unique()) + 1
    train_samples_qty = len(train_data['SessionId'].unique())
    valid_sessions = training_data['SessionId'].unique()

    active = []  # array indices active sessions in batch
    top_sess = 0  # max index on active

    while len(active) < batch_size:
        aux = data.loc[data[session_key].isin((top_sess,))]
        # search for existing session id (first one is 1)
        if len(aux) >= 2:
            active.append(top_sess)
        top_sess += 1

    ac_idx = [0 for i in range(batch_size)]  # index from where we get data, label pair
    sess_id = [i for i in range(batch_size)]  # array for fake session id
    sess_counter = batch_size

    while True:
        feats = []
        labels = []

        # check what sessions to return
        to_add_active = []
        to_add_ac_idx = []
        to_add_sess_id = []
        for sess_idx in range(len(active)):
            sess = active[sess_idx]
            # if session does not have enough data, we replace with the next one
            # (we know it will at least do for this iteration, as n >= 2 always)
            if offset_sessions[valid_sessions[sess_id[sess_idx]]] + ac_idx[sess_idx] + 1 >= offset_sessions[
                                                                                            valid_sessions[
                                                                                                sess_id[sess_idx] + 1]]:
                while True:
                    top_sess = (top_sess + 1) % train_samples_qty
                    aux = data.loc[data[session_key].isin((top_sess,))]
                    if len(aux) >= 2:
                        break

            if top_sess == 0:
                print("One epoch!")

                # print(sess)
                # print(active)
            active[sess_idx] = -1
            to_add_active.append(top_sess)
            # print(active)

            # print(ac_idx)
            ac_idx[sess_idx] = -1
            to_add_ac_idx.append(0)
            # print(ac_idx)

            sess_counter += 1
            sess_id[sess_idx] = -1
            to_add_sess_id.append(sess_counter)

            # HERE WE HAVE TO RESET MODEL GRU HIDDEN STATE

    # clean up data
    active = list(filter(lambda x: x != -1, active))
    ac_idx = list(filter(lambda x: x != -1, ac_idx))
    sess_id = list(filter(lambda x: x != -1, sess_id))
    active.extend(to_add_active)
    ac_idx.extend(to_add_ac_idx)
    sess_id.extend(to_add_sess_id)

    # get first element of all active sessions, and thats our "feat" vector
    for sess_idx in range(len(active)):
        sess = active[sess_idx]
        try:
            feat = int(data.loc[data[session_key].isin((sess,))].iloc[ac_idx[sess_idx]][mapped_item_key])
            label = int(data.loc[data[session_key].isin((sess,))].iloc[ac_idx[sess_idx] + 1][mapped_item_key])
        except:
            print("ERROR")
            print(data.loc[data[session_key].isin((sess,))])
            print(ac_idx[sess_idx])
            print(offset_sessions[sess_id[sess_idx]])
            print(offset_sessions[sess_id[sess_idx]] + ac_idx[sess_idx] + 1)
            print(offset_sessions[sess_id[sess_idx] + 1])

        ac_idx[sess_idx] += 1
        feats.append(feat)
        labels.append(label)

    feats = to_categorical(np.array(feats), n_items)
    labels = to_categorical(np.array(labels), n_items)
    # print(feats.shape)
    # print(labels.shape)

    feats.reshape(1, -1)
    labels.reshape(1, -1)
    # print(feats.shape)
    # print(labels.shape)

    yield feats, labels


tests_generator = batch_generator(training_data,
                                  training_itemidmap,
                                  training_offset_sessions,
                                  batch_size=128,
                                  fraction=1,
                                  offset=0)
next(tests_generator)

# MODEL SHOULD BE A PARAMETER FOR GRU RESETTING
def batch_generator(data, itemidmap, offset_sessions, batch_size=128, session_max_len=19, fraction=1, offset=0):
    # Eventualmente intentar volver a implementar fracciones recientes del dataset 
    item_key = 'ItemId'
    mapped_item_key = 'ItemIdx'
    session_key = 'SessionId'
    time_key = 'Time'
    n_items = len(data['ItemId'].unique())+1
    train_samples_qty = len(train_data['SessionId'].unique())
    
    active = []
    top_sess = 0
    
    while len(active) < batch_size:
        aux = data.loc[data[session_key].isin((top_sess,))]
        if len(aux) >= 2:
            active.append(top_sess) # nota: no existe la sesion 0
        top_sess += 1
        
    ac_idx = [0 for i in range(batch_size)] # indices desde donde saco par data, label
    sess_id = [i for i in range(batch_size)]
    sess_counter = batch_size

    while True:
        feats=[]
        labels=[]
        
        # check what sessions to return
        for sess_idx in range(len(active)):
            sess = active[sess_idx]
            # if session does not have enough data, we replace with the next one
            # (we know it will at least do for this iteration, as n >= 2 always)
            if offset_sessions[sess_id[sess_idx]]+ac_idx[sess_idx]+1 >= offset_sessions[sess_id[sess_idx]+1]:
                while True:
                    top_sess = (top_sess + 1) % train_samples_qty
                    aux = data.loc[data[session_key].isin((top_sess,))]
                    if len(aux) >= 2:
                        break
                        
                if top_sess == 0:
                    print("One epoch!")
                    
                #print(sess)
                #print(active)
                active.remove(sess)
                active.append(top_sess)
                #print(active)
                
                #print(ac_idx)
                del ac_idx[sess_idx]
                ac_idx.append(0)
                #print(ac_idx)
                
                sess_counter += 1
                del sess_id[sess_idx]
                sess_id.append(sess_counter)
                
                # HERE WE HAVE TO RESET MODEL GRU HIDDEN STATE
        
        # get first element of all active sessions, and thats our "feat" vector
        for sess_idx in range(len(active)):
            sess  = active[sess_idx]
            try:
                feat  = int(data.loc[data[session_key].isin((sess,))].iloc[ac_idx[sess_idx]][mapped_item_key])
                label = int(data.loc[data[session_key].isin((sess,))].iloc[ac_idx[sess_idx]+1][mapped_item_key])
            except:
                #print(data.loc[data[session_key].isin((sess,))])
                #print(ac_idx[sess_idx])
                #print(offset_sessions[sess_id[sess_idx]])
                #print(offset_sessions[sess_id[sess_idx]]+ac_idx[sess_idx]+1)
                #print(offset_sessions[sess_id[sess_idx]+1])
                pass
            ac_idx[sess_idx] += 1
            feats.append(feat)
            labels.append(label)
        
        print(ac_idx)

        feats = to_categorical(np.array(feats), n_items)
        labels = to_categorical(np.array(labels), n_items)
        #print(feats.shape)
        #print(labels.shape)
        
        feats.reshape(1, -1)
        labels.reshape(1, -1)
        #print(feats.shape)
        #print(labels.shape)
        
        yield feats, labels
        
tests_generator = batch_generator(training_data, 
                                  training_itemidmap, 
                                  training_offset_sessions, 
                                  batch_size=2, 
                                  fraction=1, 
                                  offset=0)
next(tests_generator)


# In[67]:


while True:
    next(tests_generator)


# In[63]:


# Modelo

# ToDo:
# meterle self-attention (hay implementaciones en Keras)


def custom_cosine_loss(emb):
    # y_pred ya viene con embedding, y_true solo como one-hot
    def fn(y_true, y_pred):
        y_true_emb = emb.call(y_true)[0][0]
        
        #y_true_emb = np.array([y_true], dtype='int32')
        #y_true_emb = tf.convert_to_tensor(y_true_emb)
        #y_true_emb = model.layers[0].call(y_true)
        #y_true_emb = K.get_value(y_true_emb)[0][0] # 50,
        
        return 1 - cosine_proximity(y_true_emb, y_pred)
    return fn
    
emb_size = 50
    
model = Sequential()
emb = Embedding(n_items, emb_size, input_length=19)
model.add(emb)
model.add(Dropout(0.125))
model.add(CuDNNGRU(1000)) 
model.add(Dropout(0.125))
model.add(Dense(emb_size, activation='softmax'))
#model.add(Dropout(0.2)) # Probar esto mas adelante
custom_loss = custom_cosine_loss(emb)  ## DUDA: Esta usando los pesos actuales?
model.compile(loss=custom_loss, optimizer='adam')
model.summary()

filepath="model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[ ]:


train_fraction = 1 # 1/fraction es la cantidad de sesiones mas recientes a considerar
dev_fraction = 1

offset_step=0


# In[ ]:


#todo meterle un offset de sesiones al generador para poder continuar training al cargar pesos
for epoch in range(1, 3):
    train_generator = batch_generator(train_data, batch_size=batch_size, fraction=train_fraction, offset=offset_step*epoch)
    dev_generator = batch_generator(dev_data, batch_size=batch_size, fraction=dev_fraction, offset=offset_step*epoch)
    history = model.fit_generator(train_generator,
                                steps_per_epoch=61677,#240
                                epochs=1,
                                validation_data=dev_generator,
                                validation_steps=105,
                                callbacks=callbacks_list)
    """try:
    file = ndrive.CreateFile({'title': 'model.hdf5'})
    file.SetContentFile('./model.hdf5'.format(epoch))
    file.Upload() 

    except InvalidConfigError:
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    ndrive = GoogleDrive(gauth)

    file = ndrive.CreateFile({'title': 'model.hdf5'})
    file.SetContentFile('./model.hdf5')
    file.Upload() 
    """


# In[ ]:


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
ndrive = GoogleDrive(gauth)
file = ndrive.CreateFile({'title': 'model.hdf5'})
file.SetContentFile('./model.hdf5')
file.Upload() 
get_ipython().system('ls')


# In[ ]:



# Test performance on test set

test_generator = batch_generator(test_data, batch_size=batch_size)
model.load_weights('./drive/My Drive/Cursos/2018/IIC3633/model_8.h5')
model.evaluate_generator(test_generator, steps=400, max_queue_size=10, workers=1, use_multiprocessing=False)


# In[ ]:


# Obtencion de metricas

# Paso 1: Tomar el train set, y para cada ItemId sacar su one hot y luego su embedding. Guardar esto en una matriz
# CONCLUSION: Esto ya está tal cual en la matriz de pesos de embedding. Para sacar el de un item, basta encontrar su itemidmap y luego comparar con la columna respectiva en ella
weights = model.layers[0].get_weights()[0]
print(weights.shape)


# In[ ]:


# Paso 2: Dado un embedding de output desde el modelo, obtener los k=20 vectores mas cercanos en distancia sobre el espacio de embedding

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(weights)
distances, indices = nbrs.kneighbors(weights) # Vienen ya ordenados! # Shape (37484, 20)


# In[ ]:


# Paso 3: Dado un vector embedding arbitrario, obtener el item más cercano a éste. Aplicarla sobre los 20 anteriores.
from sklearn.metrics import recall_score

test_generator = batch_generator(test_data, batch_size=batch_size)
n = 0
suma = 0
while True:
  test_batch = next(test_generator)
  pred = model.predict(test_batch[0]) # 128, 50
  label = test_batch[1]               # 128, 1


  for row_idx in range(test_batch[0].shape[0]):
    pred_row = pred[row_idx] # 50,
    label_row = label[row_idx] # 50,

    # embedding a label
    elt = np.array([label_row], dtype='int32')
    elt = tf.convert_to_tensor(elt)
    called = model.layers[0].call(elt)
    print(called.shape)
    emb_label = K.get_value(called)[0][0] # 50,

    # ahora, comparamos distancias
    label_distances, label_indices = nbrs.kneighbors(emb_label.reshape(1, -1))
    pred_distances, pred_indices = nbrs.kneighbors(pred_row.reshape(1, -1))


    # OJO: Verificar que no ocurra que uno este sobre itemidmap y el otro sobre el rango normal
    #print(label_distances)
    #print(pred_distances)
    print(label_indices)
    print(pred_indices)
    recall = recall_score(label_indices[0], pred_indices[0], average='macro')
    print(recall)
    suma += recall
    n+=1
    
print(suma/n)


# In[ ]:


# Pasar params a fn

def test2(data):
  item_key = 'ItemId'
  session_key = 'SessionId'
  time_key = 'Time'

  itemids = data[item_key].unique()
  n_items = len(itemids)

  itemidmap = pd.Series(data=np.arange(n_items), index=itemids) # Mapeo desde los 37.5k a (0, 37.5k) id
  data = pd.merge(data, pd.DataFrame({item_key:itemids, 'ItemIdx':itemidmap[itemids].values}), on=item_key, how='inner') # agrego esa columna

  for elt in indices[0]:
    print()
    
  for dist in distances:
    print(dist)
    
test2(train_data)

# Paso 4: Ya tenemos toda la informacion: el output y los 20 más cercanos a éste
# Paso 5: Calcular recall y MRR con librerías de manera sencilla (sklearn ofrece una, creo)

# LUEGO DE ESTO
# Si da muy mal comparado a M4 del paper, probar con 1000 hidden units.
# Si sigue mal, entonces entrenar el v1 por mucho tiempo, copiar los pesos de esa embedding, pegarlos aca, y entrenar de nuevo
# Si sigue mal, asumir pérdida por diferencia de implementación, y pasar a probar mecanismos de atención


# In[ ]:


# Chequeo veracidad paso 1

def test(train_data):
  item_key = 'ItemId'
  session_key = 'SessionId'
  time_key = 'Time'

  itemids = train_data[item_key].unique()
  n_items = len(itemids)

  itemidmap = pd.Series(data=np.arange(n_items), index=itemids) # Mapeo desde los 37.5k a (0, 37.5k) id
  train_data = pd.merge(train_data, pd.DataFrame({item_key:itemids, 'ItemIdx':itemidmap[itemids].values}), on=item_key, how='inner') # agrego esa columna

  for iii in range(15):
    feats = np.array([train_data['ItemIdx'].unique()[iii]], dtype='int32')
    print(feats)
    if feats.shape[0] > session_max_len:
        feats = feats[:session_max_len]
    else:
        feats = np.append(np.zeros((session_max_len-feats.shape[0],1), dtype=np.int8), feats) # left pad with zeros
    print(feats)
    feats = tf.convert_to_tensor(feats)
    print(feats)
    print(feats.shape)
    emb_elt = K.get_value(model.layers[0].call(feats))
    print(emb_elt[-1]==weights[0][iii])
  
test(train_data)

def get_train_embs(train_data, model, emb_size):
  out = np.zeros((n_items, emb_size))
  idx = 0
  #for name, values in train_data.iteritems():
  #  if name=='ItemId':
  #for elt in values:
  for elt_idx in range(len(train_data['ItemId'].unique())):
    if elt_idx % 1000 == 0:
      print(elt_idx)
    elt = np.array([train_data['ItemId'].unique()[elt_idx]], dtype='int32')
    elt = tf.convert_to_tensor(elt)
    emb_elt = K.get_value(model.layers[0].call(elt))
    print(emb_elt)
    out[idx, :] = emb_elt
    idx += 1
  print(out.shape)
  return out

emb_items = get_train_embs(train_data, model, emb_size)


# In[ ]:




