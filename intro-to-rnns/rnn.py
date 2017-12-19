import time
from collections import namedtuple

import numpy as np
import tensorflow as tf

# loading

with open('anna.txt', 'r') as f:
    text=f.read()
vocab = sorted(set(text))
print(vocab)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
print(vocab_to_int)
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

print(text[:100])

print(encoded[:100])

print(len(vocab))

def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = int(n_seqs * n_steps)
    n_batches = int(len(arr) / characters_per_batch)
    #print(characters_per_batch,n_batches)
    # Keep only enough characters to make full batches
    rlen = int(characters_per_batch * n_batches)
    arr = np.array(arr[0:rlen])
    print(arr[0])

    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs,-1))
    #print(n_batches * n_steps)
    #print(arr.shape)
    x = []
    for n in range(0, arr.shape[1], n_steps):
        x = []
        y = []
        # The features
        for i in range(n_seqs):
            seq = arr[i][n:n+n_steps]
            x.append(seq)
            y.append(seq)
        y = np.array(y)
        for i in range(n_seqs):
            y[i] = np.roll(y[i],-1)
        x = np.array(x)
        # The targets, shifted by one
        yield x, y

batches = get_batches(encoded, 10, 50)
x, y = next(batches)
print(x.shape,y.shape)
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])