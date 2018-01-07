"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
# =====================================================================================
# =====================================================================================
view_sentence_range = (0, 10)

# =====================================================================================
# =====================================================================================

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

# =====================================================================================
# =====================================================================================

import numpy as np
import problem_unittests as tests
from collections import Counter


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict....
    """
    # create dictionary with word:count
    word_counts = Counter(text)
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


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)


# =====================================================================================
# =====================================================================================
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    t = [("PERIOD", '.'),
         ("COMMA", ','),
         ("QUOTATION", '"'),
         ("EXCLAMATION", '!'),
         ("QUESTION", '?'),
         ("LEFTPAREN", '('),
         ("RIGHTPAREN", ')'),
         ("SEMICOLON", ';'),
         ("DASH", '--'),
         ("NEWLINE", '\n')
         ]
    punc = {s[1]: s[0] for s in t}
    return punc


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)
# =====================================================================================
# =====================================================================================


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
# =====================================================================================
# =====================================================================================

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
# =====================================================================================
# =====================================================================================

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# =====================================================================================
# =====================================================================================


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests
import tensorflow as tf

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # create the required placeholders
    inputs = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    rate = tf.placeholder(tf.float32, None, name="learning_rate")
    return inputs, targets, rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)


# =====================================================================================
# =====================================================================================

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # Basic LSTM cell

    # number of layers
    rnn_layers = 1

    def build_cell(rnn_size):
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        # Add dropout to the cell
        # keep_prob = 0.8  # dropout rate
        # lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return lstm

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(rnn_size) for i in range(rnn_layers)])

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

    # give a name to initial state
    initial_state = tf.identity(initial_state, "initial_state")

    return cell, initial_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)


# =====================================================================================
# =====================================================================================

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    embedding = tf.Variable(tf.random_uniform([vocab_size, embed_dim], minval=-1, maxval=1), name="embedding")
    embed = tf.nn.embedding_lookup(embedding, input_data, name="embed")
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)


# =====================================================================================
# =====================================================================================

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, "final_state")
    return outputs, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)


# =====================================================================================
# =====================================================================================

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """

    # logits
    embed = get_embed(input_data, vocab_size, embed_dim)

    # final state
    rnn_outputs, final_state = build_rnn(cell, embed)

    # output
    # use default initializers, activation_fn None specifies linear activation
    logits = tf.contrib.layers.fully_connected(rnn_outputs, vocab_size, activation_fn=None)

    return logits, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)


# =====================================================================================
# =====================================================================================

# this is my original get_batches. Being primarily  a C programmer, I think in loops
# I knew there would be a way to make it simpler and more efficient using slicing
# but I could not get that to work. see below
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    n_batches = len(int_text) // (batch_size * seq_length)
    limit = n_batches * batch_size * seq_length
    batches = []
    n = 0
    for b in range(n_batches):
        batch = []
        m = 0
        for i in range(batch_size):
            for j in range(seq_length):
                index = j + m + n
                if index < limit:
                    batch.append(int_text[index])
                else:
                    batch.append(int_text[0])
            m += seq_length * n_batches
        batches.append(batch)
        batch = []
        m = 1
        for i in range(batch_size):
            for j in range(seq_length):
                index = j + m + n
                if index < limit:
                    batch.append(int_text[index])
                else:
                    batch.append(int_text[0])
            m += seq_length * n_batches
        batches.append(batch)
        n += seq_length

    batches = np.array(batches)
    batches = batches.reshape((n_batches, 2, batch_size, seq_length))

    return batches

# I found this implementation recommeded by a mentor in the forums. it
# cleared up how simple it would be to use slices. I did check that
# the result of my original implementation and this one produced
# the same set of batches. I kept my original implementation for the test.
def get_batches_from_forum(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    batches = len(int_text) // (batch_size * seq_length)

    input_ = np.array(int_text[: (batches*batch_size*seq_length)])
    target_ = np.array(int_text[1 : (batches*batch_size*seq_length)+1])
    target_[-1] = input_[0]

    input_ = input_.reshape(batch_size, -1)
    target_ = target_.reshape(batch_size, -1)

    input_ = np.split(input_, batches, -1)
    target_ = np.split(target_, batches, -1)

    return np.array(list(zip(input_, target_)))

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)
# =====================================================================================
# =====================================================================================


# Number of Epochs
num_epochs = 2
# Batch Size
batch_size = 64
# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 300
# Sequence Length
seq_length = 100
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 100

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'
# =====================================================================================
# =====================================================================================

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
# =====================================================================================
# =====================================================================================

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches  = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
# =====================================================================================
# =====================================================================================

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
# =====================================================================================
# =====================================================================================

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()


# =====================================================================================
# =====================================================================================
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # TODO: Implement Function
    inp = loaded_graph.get_tensor_by_name("input:0")
    istate = loaded_graph.get_tensor_by_name("initial_state:0")
    fstate = loaded_graph.get_tensor_by_name("final_state:0")
    probs = loaded_graph.get_tensor_by_name("probs:0")
    return inp, istate, fstate, probs


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)


# =====================================================================================
# =====================================================================================

def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function

    index = np.random.choice(len(probabilities), p=probabilities)
    return int_to_vocab[index]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)

# =====================================================================================
# =====================================================================================

gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[dyn_seq_length - 1], int_to_vocab)

        gen_sentences.append(pred_word)

    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')

    print(tv_script)

# =====================================================================================
# =====================================================================================
