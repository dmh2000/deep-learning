import numpy as np
import tensorflow as tf
from collections import Counter


with open('../sentiment-network/reviews.txt', 'r') as f:
    reviews = f.read()
with open('../sentiment-network/labels.txt', 'r') as f:
    labels = f.read()

print(reviews[:2000])

from string import punctuation

# remove punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
# split back to reviews
reviews = all_text.split('\n')
# combine all reviews
all_text = ' '.join(reviews)
# split into words
words = all_text.split()

print(words[:100])


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict....
    """
    # create dictionary with word:count
    word_counts = Counter(words)
    # sort word_counts
    # sorted is the general sorting routine for dictionaries and other iterables
    # .get returns the value of each entry of an iterated object
    # reverse sorts in descending order
    # returns in this case a list of words sorted by frequency, highest first
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create a dictionary, keys are indices, values are words
    # word indices start at 1
    int_to_vocab = {ii + 1: word for ii, word in enumerate(sorted_vocab)}
    int_to_vocab[0] = "ERROR"
    # create a dictionary, keys are words, values are indices
    # word indices start at 1
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    vocab_to_int["ERROR"] = 0

    return vocab_to_int, int_to_vocab


# Create your dictionary that maps vocab words to integers here
vocab_to_int, int_to_vocab = create_lookup_tables(words)

# Convert the reviews to integers, same shape as reviews list, but with integers
# for each word in word list, convert to integer index into word list
reviews_ints = []
for r in reviews:
    r = r.split()
    s = [vocab_to_int[w] for w in r]
    reviews_ints.append(s)

print(words[0])
print(vocab_to_int[words[0]])
print(int_to_vocab[21024])
print(reviews_ints[:10])

# Convert labels to 1s and 0s for 'positive' and 'negative'
labels = labels.split("\n")
labels = [1 if w.lower() == "positive" else 0 for w in labels]
labels = np.array(labels, dtype=int)

review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

# Filter out that review with 0 length
reviews_ints = [r for r in reviews_ints if len(r) > 0]

seq_len = 200
features = np.empty(shape=(len(reviews_ints), seq_len), dtype=int)

for i, r in enumerate(reviews_ints):
    lr = len(r)
    if lr > seq_len:
        r = r[:200]
    elif lr < seq_len:
        pad = seq_len - lr
        r = ([0] * pad) + r
    features[i] = np.array(r)
print(features[:10, :100])

split_frac = 0.8
split_index = int(len(features) * 0.8)
train_x, val_x = features[:split_index], features[split_index:]
train_y, val_y = labels[:split_index], labels[split_index:]

split_index = int(len(val_x) * 0.5)
val_x, test_x = val_x[:split_index], val_x[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

# graph hyper parmaeters
lstm_size = 256  # number of lstm nodes in a layer
lstm_layers = 1  # number of lstm layers (1)
batch_size = 500  # input batch size
learning_rate = 0.001

n_words = len(vocab_to_int) + 1  # Adding 1 because we use 0's for padding, dictionary started at 1

# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name="input")
    labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300

# create the embedding layer
with graph.as_default():
    embedding = tf.Variable(tf.random_uniform([n_words, embed_size], minval=-1, maxval=1), name="embedding")
    embed = tf.nn.embedding_lookup(embedding, inputs_, name="embed")

# create the LSTM layer
with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    # Stack up multiple LSTM layers, for deep learning
    # don't need a loop here because there is only one LSTM layer
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

# set up dynamic rnn
with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

# create output layer, fully connected with sigmoid output
# cost is mean squared error of the predictions
with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
    # optimize by minimizing cost
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# output nodes
with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# create batches
def get_batches(x, y, batch_size=100):
    # number of batches is total input length / batch size
    n_batches = len(x) // batch_size
    # slice the input into n_batches, each of batch size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    # generate the output batches
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


# training
epochs = 10

# create a saver for checkpoints
with graph.as_default():
    saver = tf.train.Saver()

# create a session with the constructed graph
with tf.Session(graph=graph) as sess:
    # initialize global variables
    sess.run(tf.global_variables_initializer())

    # for number of epochs
    iteration = 0
    for e in range(epochs):
        # initial state for this iteration. lstm state is all zeros
        state = sess.run(initial_state)

        # enumerase the batches
        # output of enumerate is batch number, (train_x, train_y)
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            # set up the feed-dict
            feed = {inputs_: x,             # batch inputs
                    labels_: y[:, None],    # batch labels (reshaped to [:][1]
                    keep_prob: 0.5,         # keep prob for dropout
                    initial_state: state}   # lstm initial state

            # run the functions cost, final state and optimizer
            # loss is output of 'cost' function
            # state is output of 'final_state' from dynamic rnn, output of optimizer is not used
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            # every 5 iterations, print some information
            if iteration % 5 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            # every 25 itertions, execuate a validation pass
            if iteration % 25 == 0:
                val_acc = []
                # zero out the lstm state
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                # for all batches in validation data
                for x, y in get_batches(val_x, val_y, batch_size):
                    # set up feed dict
                    feed = {inputs_: x,
                            labels_: y[:,None],
                            keep_prob: 1,
                            initial_state: val_state}
                    # run the model on the validation data
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    # keep track of batch accuracy
                    val_acc.append(batch_acc)
                # print averate validation accuracy over the validation batches
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration += 1
    # save checkpints
    saver.save(sess, "checkpoints/sentiment.ckpt")

# testing
test_acc = []
with tf.Session(graph=graph) as sess:
    # restore the latest checkpoint
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    # zero out the lstm cell
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    # for each batch of test inputs
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        # set up feed dict
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        # run the model and capture the batch accurcay
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    # print averate test accuracy over the test batches
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))