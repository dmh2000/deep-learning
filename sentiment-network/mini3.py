import numpy as np


def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")


g = open('reviews.txt', 'r')  # What we know!
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()

g = open('labels.txt', 'r')  # What we WANT to know!
labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
g.close()

print(len(reviews))

print(reviews[0])

print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)

from collections import Counter

positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

for label, line in zip(labels, reviews):
    words = line.split(' ')
    for word in line.split(' '):
        if label == "POSITIVE":
            positive_counts[word] = positive_counts[word] + 1
        elif label == "NEGATIVE":
            negative_counts[word] = negative_counts[word] + 1
        else:
            print("ERROR")
        total_counts[word] = total_counts[word] + 1

print(positive_counts.most_common(5))
print(negative_counts.most_common(5))
print(total_counts.most_common(5))

# Create Counter object to store positive/negative ratios
pos_neg_ratios = Counter()

# TODO: Calculate the ratios of positive and negative uses of the most common words
#       Consider words to be "common" if they've been used at least 100 times

for word in positive_counts.keys():
    a = positive_counts[word]
    b = negative_counts[word]
    pos_neg_ratios[word] = a / (b + 1)

print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

# TODO: Convert ratios to logs
for word in pos_neg_ratios.keys():
    pos_neg_ratios[word] = np.log(pos_neg_ratios[word])

print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

print(pos_neg_ratios.most_common(5))
print(list(reversed(pos_neg_ratios.most_common()))[0:5])

# TODO: Create set named "vocab" containing all of the words from all of the reviews
vocab = set()
for line in reviews:
    words = line.split(' ')
    [vocab.add(w) for w in words]

vocab_size = len(vocab)
print(vocab_size)

# TODO: Create layer_0 matrix with dimensions 1 by vocab_size, initially filled with zeros
layer_0 = np.zeros((1, len(vocab)))

print(layer_0.shape)

# Create a dictionary of words in the vocabulary mapped to index positions
# (to be used in layer_0)
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

# display the map of words to indices
print(word2index)


def update_input_layer(review):
    """ Modify the global layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
        review(string) - the string of the review
    Returns:
        None
    """
    global layer_0
    # clear out previous state by resetting the layer to be all 0s
    layer_0 *= 0

    # TODO: count how many times each word is used in the given review and store the results in layer_0
    for word in review.split(' '):
        layer_0[0, word2index[word]] += 1


update_input_layer(reviews[0])
print(layer_0)


def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    # TODO: Your code here
    return 1 if label == "POSITIVE" else 0


print(labels[0])
print(get_target_for_label(labels[0]))
print(labels[1])
print(get_target_for_label(labels[1]))

import time
import sys
import numpy as np


# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)

        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):

        review_vocab = set()
        # populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words
        #       using "split(' ')" instead of "split()".
        for review in reviews:
            for word in review.split(' '):
                review_vocab.add(word)

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

        label_vocab = set()
        # populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        for label in labels:
            label_vocab.add(label)

        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)

        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # do the same thing you did for self.word2index and self.review_vocab,
        #       but for self.label2index and self.label_vocab instead
        for i, word in enumerate(self.label_vocab):
            self.label2index[word] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        # initialize self.weights_1_2 as a matrix of random values.
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.rand(hidden_nodes, output_nodes) - 0.5

        # Create the input layer, a two-dimensional matrix with shape
        #       1 x input_nodes, with all values initialized to zero
        self.layer_0 = np.zeros((1, input_nodes))

    def update_input_layer(self, review):
        # You can copy most of the code you wrote for update_input_layer
        #       earlier in this notebook.
        #
        #       However, MAKE SURE YOU CHANGE ALL VARIABLES TO REFERENCE
        #       THE VERSIONS STORED IN THIS OBJECT, NOT THE GLOBAL OBJECTS.
        #       For example, replace "layer_0 *= 0" with "self.layer_0 *= 0"
        self.layer_0 *= 0.0
        for word in review.split(' '):
            if word in self.word2index.keys():
                self.layer_0[0][self.word2index[word]] += 1

    def get_target_for_label(self, label):
        # Copy the code you wrote for get_target_for_label
        #       earlier in this notebook.
        return 1 if label == "POSITIVE" else 0

    def sigmoid(self, x):
        # Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        # Return the derivative of the sigmoid activation function,
        #       where "output" is the original output from the sigmoid fucntion
        return output * (1.0 - output)

    def train(self, training_reviews, training_labels):

        # make sure out we have a matching number of reviews and labels
        assert (len(training_reviews) == len(training_labels))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            # TODO: Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]

            # TODO: Implement the forward pass through the network.
            #       That means use the given review to update the input layer,
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            #
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            # sum over weights
            review = review.lower()
            self.update_input_layer(review)

            hidden_layer = self.layer_0.dot(self.weights_0_1)

            output_layer = self.sigmoid(hidden_layer.dot(self.weights_1_2))

            # TODO: Implement the back propagation pass here.
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you
            #       learned in class.
            # Output error (output gradient)
            output_error = self.get_target_for_label(label) - output_layer  # Output layer error is the difference between desired target and actual output.

            # output error term
            output_error_term = output_error * self.sigmoid_output_2_derivative(output_layer)

            # hidden error
            hidden_error = output_error_term.dot(self.weights_1_2.T)

            # hidden error term
            hidden_error_term = hidden_error

            # Weight step (hidden to output)
            self.weights_1_2 += hidden_layer.T.dot(output_error_term) * self.learning_rate

            # Weight step (input to hidden)
            self.weights_0_1 += self.layer_0.T.dot(hidden_error_term) * self.learning_rate

            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error
            #       is less than 0.5. If so, add one to the correct_so_far count.
            if (output_layer >= 0.5) and (label == "POSITIVE"):
                correct_so_far += 1
            elif (output_layer < 0.5) and (label == "NEGATIVE"):
                correct_so_far += 1
            else:
                correct_so_far += 0

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i + 1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")
            if (i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """

        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if pred == testing_labels[i]:
                correct += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i + 1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i + 1))[:4] + "%")

    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # TODO: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction
        #             might come from anywhere, so you should convert it
        #             to lower case prior to using it.
        # sum over weights
        review = review.lower()
        self.update_input_layer(review)

        hidden_layer = self.layer_0.dot(self.weights_0_1)

        output_layer = self.sigmoid(hidden_layer.dot(self.weights_1_2))

        # TODO: The output layer should now contain a prediction.
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`,
        #       and `NEGATIVE` otherwise.
        return "POSITIVE" if output_layer[0] >= 0.5 else "NEGATIVE"


mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.001)

mlp.test(reviews[-1000:], labels[-1000:])

# mlp.train(reviews[:-1000],labels[:-1000])

mlp.train(reviews[:-1000], labels[:-1000])
