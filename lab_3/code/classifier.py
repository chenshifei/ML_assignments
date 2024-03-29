from time import time
import collections
import torch
import torch.optim as optim
import numpy

import util2

##################################################
# Welcome to Lab 3!                              #
#                                                #
# Here we get to understand PyTorch,             #
# which hopefully will make many things          #
# easier.  It is also your first step            #
# towards building more elaborate classifiers,   #
# and eventually neural networks.                #
#                                                #
# Happy programming!                             #
#                                                #
#                                      Andy      #
##################################################

seed = 1

torch.manual_seed(seed) # Fixes random seed for replicability
numpy.random.seed(seed)


# 1. Forward function
# This is the class for the linear classifier model.
# We will train and update this.
# You need to define the forward function: What it does when called.
class Net(torch.nn.Module):
    """Wraps the weights and bias in a Module so that we can perform easy
    backpropagation and optimisation on them

    Required parameters on instantiation:
    --nfeatures   : The number of features in each line (only 1D for this task).
                    No default.
    --initialiser : How the feature weights should be initialised.
                    Zeros or normal (mu = 1, sd = 1).
                    Default: zeros
    """

    def __init__(self, nfeatures, initialiser='zero'):
        super(Net, self).__init__()
        if initialiser == 'zero':
            self.weights = torch.zeros([nfeatures], dtype=torch.float)
            self.bias = torch.zeros([1], dtype=torch.float)
        elif initialiser == 'normal':
            self.weights = 0.1 * torch.randn([nfeatures])
            self.bias = 0.1 * torch.randn([1])

        self.weights = torch.nn.Parameter(self.weights)  # Parametrise weights
        self.bias = torch.nn.Parameter(self.bias)  # Parametrise bias

    def forward(self, x):
        # YOUR CODE HERE!
        # Define the forward function of the classifier
        # Its expected input x is a tensor of the shape m*n, where n is the
        # number of features in the data and m is the number of examples.
        return torch.matmul(x, self.weights) + self.bias




# 2. Autograd and optimisation
# This training loop is calling the loss function and computing loss.
# You'll notice that nothing is actually happening, though.
# Find out how to automatically backpropagate loss and perform
# the gradient descent step, and implement this.
def train(train_x, train_y, nfeatures,
          val_x, val_y,
          loss_function,
          l_rate=0.1, weight_decay=0.1,
          patience=-1, patience_metric='val_loss',
          n_epoch=20, verbose=1):

    net = Net(nfeatures, initialiser='zero')  # Here we instantiate our Net

    # 'Criterion' is how loss function is referred to in PyTorch tutorials.  Don't ask me why.
    criterion = loss_function

    # optimizer = optim.SGD(net.parameters(), lr=l_rate, weight_decay=weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=l_rate, weight_decay=weight_decay)

    training_log = []

    if verbose:
        print('Beginning training.')

    n = 0
    last_patience_metric = 0
    if patience_metric == 'val_loss':
        last_patience_metric = 1
    epoches_after_patience = 0
    best_model = net

    while n < n_epoch:

        epoch_start = time()

        for batch_x, batch_y in zip(train_x, train_y):

            margins = net(batch_x)
            val_margins = net(val_x)

            # Calculates training loss
            training_loss = criterion(margins, batch_y)

            # Calculates validation loss
            with torch.no_grad():  # We don't want to backprop this!
                validation_loss = criterion(val_margins, val_y)

            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            training_loss.backward()
            optimizer.step()    # Does the update

        epoch_end = time()


        # This part generates statistics, graphs and stuff
        train_predictions = predict(net, batch_x)
        train_accuracy = accuracy(train_predictions, batch_y)
        _, _, train_F1 = precision_recall_F1(train_predictions, batch_y)

        val_predictions = predict(net, val_x)
        val_accuracy = accuracy(val_predictions, val_y)
        val_precision, val_recall, val_F1 = precision_recall_F1(val_predictions, val_y)

        log_record = collections.OrderedDict()
        log_record['training_acc'] = train_accuracy
        log_record['training_loss'] = training_loss.item()
        log_record['training_F1'] = train_F1
        log_record['val_acc'] = val_accuracy
        log_record['val_loss'] = validation_loss.item()
        log_record['val_F1'] = val_F1
        log_record['val_pre'] = val_precision
        log_record['val_recall'] = val_recall
        log_record['epoch_time'] = epoch_end - epoch_start

        training_log.append(log_record)

        if verbose:
            util2.display_log_record(n, log_record)

        if patience > -1:
            metric_diff = 0
            if patience_metric == 'val_loss':
                metric_diff = abs(last_patience_metric - validation_loss.item())
                last_patience_metric = validation_loss.item()
            elif patience_metric == 'val_acc':
                metric_diff = abs(val_accuracy - last_patience_metric)
                last_patience_metric = val_accuracy

            if metric_diff < 0.0001:
                epoches_after_patience += 1
                best_model = net
            else:
                epoches_after_patience = 0

            if epoches_after_patience == patience:
                return best_model, training_log

        n += 1

    if verbose:
        print('Training complete.')

    # output of net
    return net, training_log


# Prediction and accuracy.
# You don't need to worry about these (unless for some reason
# you want to fiddle with the predict threshold).
####################################################################
def predict(net, sd_data):
    """Applies the classifier defined by the weights and the bias to the data
    and returns a list of predicted labels."""
    margin = net(sd_data)
    predictions = torch.where(margin > 0.0,
                              torch.FloatTensor([1]),
                              torch.FloatTensor([0]))
    return predictions


def accuracy(predictions, labels):
    """Computes an accuracy score given prediction and label vectors"""
    return predictions[predictions == labels].numel() / labels.numel()

def precision_recall_F1(predictions, labels):
    """Computes P, R, F1 given prediction and label vectors"""
    pred_pos = predictions == 1
    true_pos = labels == 1
    correct_pred = pred_pos[pred_pos == true_pos]
    correct_pred = correct_pred[correct_pred == 1].numel()
    total_true = true_pos[true_pos == 1].numel()
    total_pred = pred_pos[pred_pos == 1].numel()

    try:
        recall = correct_pred / total_true
        precision = correct_pred / total_pred
        F1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        recall = 0.0
        precision = 0.0
        F1 = 0.0

    return precision, recall, F1


#################################################################


# 3. Loss Functions
# Below are some very basic custom loss functions.
# They're the same ones as we used in Lab 2 (Log Loss and Hinge Loss.
# Implement these functions using PyTorch language.
class LogisticLoss:
    @staticmethod
    def __call__(pred, y, reduction='mean'):
        # Your code here!
        # Compute logistic loss over the whole tensor of weights.
        # Reduction should be mean reduction rather than log reduction:
        # (loss / |pred|) rather than (loss / log(2))

        loss = (torch.ones_like(pred) + (-pred * y).exp()).log().sum()

        # I took a look at the PyTorch source code
        # https://github.com/pytorch/pytorch/blob/master/aten/src/THNN/generic/SoftMarginCriterion.c
        # and realized the meaning of 'reduction'
        if reduction == 'mean':
            loss = loss / torch.numel(pred)
        return loss

class HingeLoss:
    @staticmethod
    def __call__(pred, y, hinge=0):
        # Your code here!
        # Compute hinge loss over the whole tensor of weights.
        loss = (torch.ones_like(pred) - pred * y).clamp(min=hinge).sum()
        return loss / torch.numel(pred)


def main():
    """The main training pipeline."""
    # 4. Tuning the model
    # Choose your task here:
    # Detect comment with personal attack
    #  (baseline accuracy: 91%)
    # Detect conversation with personal attack
    #  (baseline accuracy: 65%)
    # Choose one and comment out the other
    chosen_task = 'comment'
    #chosen_task = 'conversation'

    # Regularisation strength
    # This is also known as "weight decay", particularly in some optimisation
    # algorithms.  PyTorch uses this terminology.
    weight_decay = 0.0001

    # Learning rate
    learning_rate = 0.1

    # Number of training epochs
    epochs = 300

    batch_size = 1

    patience = 30
    patience_metric = 'val_acc'

    # Loss function to use (select one and comment out the other)
    # loss_function = torch.nn.SoftMarginLoss()
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([2.2]))
    # loss_function = LogisticLoss()
    # loss_function = HingeLoss()
    # loss_function = torch.nn.MSELoss()


    # 5. Only enable once you're done with tuning
    enable_test_set_scoring = False

    enable_plots = True

    # Type of features to use. This can be set to 'bigram' or 'unigram+bigram' to use
    # bigram features instead of or in addition to unigram features.
    # Not required for assignment.
    feature_type = 'unigram'

    # END OF HYPERPARAMETERS

    # Load the data.
    print()
    print('===================')
    print('CLASSIFIER TRAINING')
    print('===================')
    print()
    print('Loading data sets...')


    data_dir = 'tokenised_conversations.json'
    data = util2.load_awry_data(data_dir, task=chosen_task)

    data.select_feature_type(feature_type)

    # Split the data set randomly into training, validation and test sets.
    training_data, val_data, test_data = data.train_val_test_split()
    nfeatures = len(training_data.vocabulary)
    mini_batches = training_data.random_split(range(1, batch_size + 1))

    print('Loaded and split data into train/val/test...')
    training_labels = [torch.FloatTensor(b.labels) for b in mini_batches]
    val_labels = torch.FloatTensor(val_data.labels)
    test_labels = torch.FloatTensor(test_data.labels)

    print('Converting datasets to dense representation...')

    # Convert to dense representation
    ds_train = [util2.sparse_to_dense(b, nfeatures) for b in mini_batches]
    print('Train converted...')
    ds_val = util2.sparse_to_dense(val_data, nfeatures)
    print('Val converted...')
    ds_test = util2.sparse_to_dense(test_data, nfeatures)
    print('Test converted...')
    # And convert to torch Tensors
    ds_train = [torch.FloatTensor(t) for t in ds_train]
    ds_val = torch.FloatTensor(ds_val)
    ds_test = torch.FloatTensor(ds_test)

    print('Data sets loaded.\n')




    # Begin training model
    model, training_log = train(train_x=ds_train,
                                train_y=training_labels,
                                nfeatures=nfeatures,
                                val_x=ds_val,
                                val_y=val_labels,
                                loss_function=loss_function,
                                l_rate=learning_rate, weight_decay=weight_decay,
                                patience=patience, patience_metric=patience_metric,
                                n_epoch=epochs, verbose=1)

    # Show statistics - you can change top N to another positive integer value
    top_N = 10
    title = 'Model performance data'
    util2.show_stats(
        title,
        training_log,
        model.weights,
        model.bias,
        training_data.vocabulary,
        top_N)

    # If you want to plot accuracy instead of loss, change
    # training_loss -> training_acc ; val_loss -> val_acc
    if enable_plots:
        util2.create_plots(
            title,
            training_log,
            model.weights,
            log_keys=(
                'training_loss',
                'val_loss'))

    if enable_test_set_scoring:

        # Probably not necessary, but again this avoids backpropagating test
        # loss
        with torch.no_grad():
            test_loss = loss_function(model(ds_test), test_labels).item()

        test_predictions = predict(model, ds_test)
        test_accuracy = accuracy(test_predictions, test_labels)
        _, _, test_F1 = precision_recall_F1(test_predictions, test_labels)

        print()
        print('Test set performance:')
        print()
        print('Test accuracy:', test_accuracy)
        print('Test F1:', test_F1)
        print('Test loss: {0:.3}'.format(test_loss))


if __name__ == '__main__':
    main()
