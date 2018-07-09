import pickle
import torch


i2c = pickle.load(open('./data/i2c', 'rb'))

def sampling(inputs, outputs):
    for i in inputs:
        print(i2c[i], end='')
    print(' -> ')
    for i in outputs:
        print(i2c[i], end='')
    print()
