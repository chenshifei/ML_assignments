import collections
import math

import util


def main():
    # TRAINING HYPERPARAMETERS
    # Modify the following lines to change the training hyperparameters.

    # Regularisation strength
    reg_lambda = 0.001

    # Learning rate
    learning_rate = 0.001

    # Number of training iterations
    niterations = 5

    # Loss function to use (select one and comment out the other)
    loss_function = LogisticLoss()
    # loss_function = HingeLoss()

    # Type of regularisation to use (select one and comment out the other)
    regulariser = L1Regulariser()
    # regulariser = L2Regulariser()

    # This should only be enabled once you've decided on a final set of hyperparameters
    enable_test_set_scoring = False

    # Type of features to use. This can be set to 'bigram' or 'unigram+bigram' to use
    # bigram features instead of or in addition to unigram features.
    # Not required for assignment.
    feature_type = 'unigram'

    # END OF HYPERPARAMETERS

    # First test the parts to be implemented and warn if something's wrong.
    print('=============')
    print('SANITY CHECKS')
    print('=============')
    print()

    util.run_tests()

    # Load the data.

    print()
    print('===================')
    print('CLASSIFIER TRAINING')
    print('===================')
    print()
    print('Loading data sets...')

    data_dir = './poldata/poldata.zip'
    data = util.load_movie_data(data_dir)

    data.select_feature_type(feature_type)

    # Split the data set randomly into training, validation and test sets.
    training_data, val_data, test_data = data.train_val_test_split()

    # Train the classifier.
    print('Starting training.')
    weights, bias, training_log = train(training_data, val_data,
                                        loss_function, regulariser,
                                        reg_lambda, learning_rate, niterations)
    print('Training completed.')

    print()
    print('=====================')
    print('MODEL CHARACTERISTICS')
    print('=====================')
    print()

    # Display some useful statistics about the model and the training process.
    title = 'Data set: %s - Regulariser: %g - Learning rate: %g' % (data.name, reg_lambda, learning_rate)

    print()
    util.show_stats(title, training_log, weights, bias, data.vocabulary, top_n=20)
    util.create_plots(title, training_log, weights, log_keys=['training_loss_reg', 'val_loss'])

    if enable_test_set_scoring:
        # Check the performance on the test set.
        test_loss = loss_function.unregularised_loss(weights, bias, test_data)
        test_predictions = predict(weights, bias, test_data)
        test_accuracy = accuracy(test_data.labels, test_predictions)

        print()
        print('====================')
        print('TEST SET PERFORMANCE')
        print('====================')
        print()
        print('Test loss: %g' % test_loss)
        print('Test accuracy: %g' % test_accuracy)


def predict(weights, bias, data):
    """Applies the classifier defined by the weights and the bias to the data and returns a list of predicted labels."""
    predictions = []
    for X in data.features():
        margin = sum(weights[feature] for feature in X) + bias
        if margin > 0:
            predictions.append(1.0)
        else:
            predictions.append(-1.0)

    return predictions


def train(training_data, val_data, loss_fn, regulariser, reg_lambda, learning_rate, niterations):
    """Trains a linear classifier.

    Arguments:
        training_data -- the training data set
        val_data -- the validation data set
        loss_fn -- an object implementing the loss function to use
        regulariser -- an object implementing the regulariser to use
        reg_lambda -- the regularisation strength (lambda)
        learning_rate -- the learning rate for gradient descent
        niterations -- number of iterations to run."""

    log = []
    nfeatures = len(training_data.vocabulary)

    # Initialise weights and bias to 0
    weights = [0.0 for _ in range(nfeatures)]
    bias = 0.0

    # Training loop
    for i in range(niterations):
        # Compute the gradients of the loss function and the regulariser separately
        loss_grads, bias_grad = loss_fn.gradients(weights, bias, training_data)
        reg_grads = regulariser.gradients(weights)

        # Add the gradients, weighting the regulariser gradient by lambda.
        weight_grads = [x + reg_lambda * y for x, y in zip(loss_grads, reg_grads)]

        # Perform the gradient descent update on the weights and the bias.
        weights, bias = gradient_descent_step(learning_rate,
                                              weights, bias,
                                              weight_grads, bias_grad)

        # Now we're done.
        # The rest of the loop just computes some useful info to monitor the training progress,
        # but it's not required for the algorithm to work.

        # Training and validation loss
        training_loss_unreg = loss_fn.unregularised_loss(weights, bias, training_data)
        training_loss_reg = training_loss_unreg + reg_lambda * regulariser.loss(weights)
        val_loss = loss_fn.unregularised_loss(weights, bias, val_data)

        # Size of the weights and the gradients
        param_norm = l2_norm(weights + [bias])
        grad_norm = l2_norm(weight_grads + [bias_grad])

        # Training and validation accuracy
        training_predictions = predict(weights, bias, training_data)
        training_accuracy = accuracy(training_data.labels, training_predictions)

        val_predictions = predict(weights, bias, val_data)
        val_accuracy = accuracy(val_data.labels, val_predictions)

        # All of these values are stored in a dictionary.
        log_record = collections.OrderedDict()
        log_record['training_loss_reg'] = training_loss_reg
        log_record['training_loss_unreg'] = training_loss_unreg
        log_record['training_acc'] = training_accuracy
        log_record['val_loss'] = val_loss
        log_record['val_acc'] = val_accuracy
        log_record['param_norm'] = param_norm
        log_record['grad_norm'] = grad_norm

        # Display info on training progress
        util.display_log_record(i, log_record)
        log.append(log_record)

    return weights, bias, log


def gradient_descent_step(learning_rate, old_weights, old_bias, weight_grads, bias_grad):
    """Performs a gradient descent update on the weights and the bias and returns the new values."""
    ### YOUR CODE HERE ###
    # Calculate the weight vector and the bias after a single gradient descent update
    # given the learning rate and gradients and return the new weights and bias.
    # Until you've implemented this correctly, we just return the old weights and bias
    # without updating.
    new_weights = old_weights
    new_bias = old_bias

    return new_weights, new_bias


def l2_norm(vector):
    """Computes the l2 norm of a vector."""
    return math.sqrt(sum(x * x for x in vector))


def accuracy(gold, hypothesis):
    """Computes an accuracy score given two vectors of labels."""
    return sum(g == h for g, h in zip(gold, hypothesis)) / len(gold)


class LogisticLoss:
    @staticmethod
    def unregularised_loss(weights, bias, data):
        """Computes the unregularised loss of the linear classifier on the provided data set."""
        loss = 0.0
        for X, y in zip(data.features(), data.labels):
            margin = sum(weights[feature] for feature in X) + bias
            loss += math.log(1.0 + math.exp(-y * margin))

        loss /= math.log(2)

        # Divide the loss by the number of examples to make it independent of the data size.
        loss /= len(data)
        return loss

    @staticmethod
    def gradients(weights, bias, data):
        """Computes the gradients of the loss function with respect to the weights and bias."""
        weight_gradients = [0.0 for _ in weights]
        bias_gradient = 0.0
        for X, y in zip(data.features(), data.labels):
            margin = sum(weights[feature] for feature in X) + bias
            bias_gradient += -1.0 / math.log(2) * y / (1.0 + math.exp(y * margin))
            for feature in X:
                weight_gradients[feature] += -1.0 / math.log(2) * y / (1.0 + math.exp(y * margin))

        # Divide the loss by the number of examples to make it independent of the data size.
        weight_gradients_per_example = [g / len(data) for g in weight_gradients]
        bias_gradient_per_example = bias_gradient / len(data)

        return weight_gradients_per_example, bias_gradient_per_example


class HingeLoss:
    @staticmethod
    def unregularised_loss(weights, bias, data):
        """Computes the unregularised loss of the linear classifier on the provided data set."""
        loss = 0.0

        ### YOUR CODE HERE ###
        # Compute the hinge loss of the data given weights and bias

        # Divide the loss by the number of examples to make it independent of the data size.
        loss /= len(data)
        return loss

    @staticmethod
    def gradients(weights, bias, data):
        """Computes the gradients of the loss function with respect to the weights and bias."""
        weight_gradients = [0.0 for _ in weights]
        bias_gradient = 0.0

        ### YOUR CODE HERE ###
        # Compute the gradients of the hinge loss with respect to the weights and bias

        # Divide the loss by the number of examples to make it independent of the data size.
        weight_gradients_per_example = [g / len(data) for g in weight_gradients]
        bias_gradient_per_example = bias_gradient / len(data)

        return weight_gradients_per_example, bias_gradient_per_example


class L1Regulariser:
    @staticmethod
    def loss(weights):
        """Computes the loss term added by the regulariser (a scalar)."""
        return sum(abs(w) for w in weights)

    @staticmethod
    def gradients(weights):
        """Computes subgradients of the regularisation term with respect to the weights (a vector)."""
        return [1.0 if w > 0 else -1.0 for w in weights]


class L2Regulariser:
    @staticmethod
    def loss(weights):
        """Computes the loss term added by the regulariser (a scalar)."""
        ### YOUR CODE HERE ###
        # Compute the value of the l2 regularisation term.
        reg_loss = 0.0

        return reg_loss

    @staticmethod
    def gradients(weights):
        """Computes the gradients of the regularisation term with respect to the weights (a vector)."""
        ### YOUR CODE HERE ###
        # Compute the correct gradient vector corresponding to the l2 loss term.
        # The existing code just returns a vector of zeros. This should be replaced.
        reg_gradient = [0.0 for _ in weights]

        return reg_gradient


if __name__ == '__main__':
    main()
