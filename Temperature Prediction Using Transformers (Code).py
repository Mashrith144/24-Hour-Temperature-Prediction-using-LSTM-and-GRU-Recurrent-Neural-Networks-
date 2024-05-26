#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


def loadData():
    df = pd.read_csv("C:/Users/rmrco/Desktop/jena_climate_2009_2016.csv")
    df.drop("Date Time",axis=1,inplace=True)
    temperatures = np.array(df["T (degC)"])
    raw_data = np.array(df)
    
    return raw_data, temperatures

raw_data, temperatures = loadData()


# In[3]:


def normaliseData(raw_data,normalise_limit):
    mean = raw_data[:normalise_limit].mean(axis=0)
    raw_data -= mean
    std = raw_data[:normalise_limit].std(axis=0)
    raw_data /= std
    return raw_data,std,mean

raw_data,std,mean = normaliseData(raw_data,train_sample_count)


# In[4]:


train_sample_count = int(len(raw_data)*0.5)
val_sample_count = int(len(raw_data)*0.25)
test_sample_count = len(raw_data) - (train_sample_count + val_sample_count)

sampling_rate = 6
sequence_length = 120 
delay = 6* (sequence_length + 24 -1)
batch_size = 256

def getDataGenerators(raw_data,temperatures,sampling_rate,sequence_length,delay,batch_size):
    train_dataset = keras.utils.timeseries_dataset_from_array(
    data = raw_data[:-delay],
    targets = temperatures[delay:],
    sequence_length = sequence_length,
    sampling_rate=sampling_rate,
    batch_size=batch_size,
    shuffle=True,
    start_index=0,
    end_index=train_sample_count,
    )

    val_dataset = keras.utils.timeseries_dataset_from_array(
        data = raw_data[:-delay],
        targets = temperatures[delay:],
        sequence_length = sequence_length,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        shuffle=True,
        start_index=train_sample_count,
        end_index=train_sample_count + val_sample_count,
    )

    test_dataset = keras.utils.timeseries_dataset_from_array(
        data = raw_data[:-delay],
        targets = temperatures[delay:],
        sequence_length = sequence_length,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        shuffle=True,
        start_index=val_sample_count
    )
    
    return train_dataset, val_dataset, test_dataset
    
train_dataset, val_dataset, test_dataset = getDataGenerators(raw_data,
                                                             temperatures,
                                                             sampling_rate,
                                                             sequence_length,
                                                             delay,
                                                             batch_size)


# In[5]:


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = keras.layers.Dropout(dropout)(x)
    res = x + inputs

    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


# In[6]:


def build_model(input_shape,head_size,num_heads,ff_dim,num_transformer_blocks,dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    outputs = Dense(1)(x)

    return keras.Model(inputs, outputs)


# In[7]:


model = build_model(
    input_shape = (sequence_length, raw_data.shape[-1]),
    head_size=256,
    num_heads=1,
    ff_dim=4,
    num_transformer_blocks=2,
    dropout=0.25,
)

model.compile(
    loss="mse",
    optimizer="rmsprop",
    metrics=["mae"],
)

model.fit(train_dataset,
          epochs=20,
          validation_data=val_dataset,
          batch_size=32)
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

