import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # use sigmoid activation
        self.activation_function = lambda x: 1.0 / (1.0 + np.exp(-x))  # Replace 0 with your sigmoid calculation.

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here

            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        # =========================
        #  compute the hidden layer
        # =========================

        # sum over weights
        hidden_inputs = X.dot(self.weights_input_to_hidden)
        # reshape to (n,1)
        # hidden_inputs = hidden_inputs.reshape((self.hidden_nodes, 1))
        # apply the activation function
        hidden_outputs = self.activation_function(hidden_inputs)

        # =====================
        #  compute output layer
        # =====================
        # multiply by weights
        final_outputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # Output error (output gradient)
        output_error = y - final_outputs  # Output layer error is the difference between desired target and actual output.

        # output error term
        # output is a sum, not sigmoid, so derivative is 1
        output_error_term = output_error

        # hidden error
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)

        # hidden error term
        hidden_prime = hidden_outputs * (1.0 - hidden_outputs)
        hidden_error_term = hidden_error * hidden_prime

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]

        # Weight step (hidden to output)
        delta_weights_h_o += (hidden_outputs * output_error_term).reshape((self.hidden_nodes, 1))

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += (self.lr * delta_weights_h_o) / n_records  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += (self.lr * delta_weights_i_h) / n_records  # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''

        # =========================
        # compute the hidden layer
        # =========================
        # sum over weights
        hidden_inputs = features.dot(self.weights_input_to_hidden)

        # apply the activation function
        hidden_outputs = self.activation_function(hidden_inputs)

        # =========================
        # compute the output layer
        # =========================
        # sum over the weights
        final_outputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 3500
learning_rate = 0.5
hidden_nodes = 24
output_nodes = 1
