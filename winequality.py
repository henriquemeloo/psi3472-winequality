# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import Adam

## Gerando base de dados
td = pd.read_csv('./database/winequality-red.csv', sep=';', decimal='.')
# Gerando conjuntos de treino e teste
msk = np.random.rand(len(td)) < 0.85
train_data = td[msk]
test_data = td[~msk]
print('{len_train} elements in train and {len_test} elements in test'.format(
    len_train=len(train_data), len_test=len(test_data))
)
# Variaveis de entrada e de saida
inputs = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]
output = 'quality'

## Treinando modelo
# define the keras model
model = Sequential()
model.add(Dense(11, input_dim=11, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='linear'))
# compile the keras model
adam = Adam(lr=5e-2)
model.compile(loss='mean_squared_error', optimizer='adam')
# fit the keras model on the dataset
history = model.fit(
    x=train_data[inputs],
    y=train_data[output],
    epochs=600,
    batch_size=120,
    validation_split=.15,
    verbose=True
)

## Avaliando resultado
# evaluate model
train_rms = np.sqrt(history.history['loss'][-1])
val_rms = np.sqrt(history.history['val_loss'][-1])
test_rms = np.sqrt(
    model.evaluate(x=test_data[inputs], y=test_data[output], verbose=False)
)
print("""
RMS de treino: {train_rms}
RMS de validacao: {val_rms}
RMS de teste: {test_rms}
    """.format(
    train_rms=train_rms,
    val_rms=val_rms,
    test_rms=test_rms
))
# Plotando resultado
plt.rcParams.update({'font.size':14})
plt.figure(figsize=(9,9))
plt.plot(np.sqrt(history.history['loss']))
plt.plot(np.sqrt(history.history['val_loss']))
plt.legend([u'RMS de treino', u'RMS de validação'])
plt.ylabel('RMS')
plt.xlabel(u'Época de treinamento')
plt.title(u'Progressão do treinamento da rede neural wine_quality')
plt.savefig('./images/winequality_rms_train_val.png')
