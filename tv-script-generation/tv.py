"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests
from collections import Counter

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]

view_sentence_range = (0, 10)

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


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)


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

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

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
    # TODO: Implement Function
    inputs = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    rate = tf.placeholder(tf.float32, None, name="learning_rate")
    return inputs, targets, rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # Basic LSTM cell

    # number of layers
    rnn_layers = 2

    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    # Add dropout to the cell
    keep_prob = 1.0  # no dropout to start with
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * rnn_layers)

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

    # give a name to initial state
    initial_state = tf.identity(initial_state, "initial_state")

    return cell, initial_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)


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
    print("build-rnn " + str(final_state.get_shape()))
    return outputs, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)


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
    print(embed.get_shape())

    # final state
    rnn_outputs, final_state = build_rnn(cell, embed)
    print(rnn_outputs.get_shape())
    print(final_state.get_shape())

    logits = tf.contrib.layers.fully_connected(rnn_outputs, vocab_size, activation_fn=tf.sigmoid)

    print(logits.get_shape())

    return logits, final_state


print("testing----")
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)


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


x = 0
print(np.array(range(x * 35, x * 35 + 5)))
print(np.array(range(x * 35 + 1, x * 35 + 1 + 5)))

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)
