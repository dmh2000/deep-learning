import numpy as np


# 3x2

a = [
    [[1, 2], [7, 8], [13, 14]],
    [[2, 3], [8, 9], [14, 15]]]

# 3 x 3
b = [
    [
        [[1, 2, 3], [10, 11, 12], [19, 20, 21]],
        [[2, 3, 4], [11, 12, 13], [20, 21, 22]]
    ],
    [
        [[4, 5, 6], [13, 14, 15], [21, 22, 23]],
        [[5, 6, 7], [14, 15, 16], [22, 23, 1]]
    ]
]


def get_batches(input, batch_size, seq_len):
    n = len(input) // (batch_size * seq_len)
    print(n)
    m = 0
    batch = []
    for b in range( n):
        j = 0
        b1 = []
        for k in range(batch_size):
            seq = []
            for i in range(seq_len):
                index = m + i + j
                seq.append(input[index])
            j += seq_len
            b1.append(seq)
        j = 1
        b2 = []
        for k in range(batch_size):
            seq = []
            for i in range(seq_len):
                index = m + i + j
                if index < len(input):
                    seq.append(input[m + i + j])
                else:
                    seq.append(input[0])
            j += seq_len
            b2.append(seq)
        batch.append([b1,b2])
        m += seq_len * batch_size
    return batch


input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
b = get_batches(input, 3, 2)
print(b)
#
#input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#get_batches(input, 3, 2)
#
# input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# get_batches(input, 3, 2)
#
# input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# get_batches(input, 3, 2)
#
input = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
b = get_batches(input, 3, 3)
print(b)