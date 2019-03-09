import collections
import glob
import itertools
import matplotlib.pyplot as pyplot
import numpy
import os
import sys
import json
import re

from nltk.corpus import stopwords


#import classifier


if sys.version_info[0] != 3:
    print('Please use Python 3 to run this code.')
    sys.exit(1)



#Stats and graphs
def show_stats(title, log, weights, bias, vocabulary, top_n=10):
    print(title)
    print()

    best_training_loss = min(l['training_loss'] for l in log)
    best_validation_loss = min(l['val_loss'] for l in log)

    best_training_accuracy = max(l['training_acc'] for l in log)
    best_validation_accuracy = max(l['val_acc'] for l in log)

    best_training_F1 = max(l['training_F1'] for l in log)
    best_validation_F1 = max(l['val_F1'] for l in log)

    print('Best training loss: %g' % best_training_loss)
    print('Final training loss: %g' % log[-1]['training_loss'])
    print('Best validation loss: %g' % best_validation_loss)
    print('Final validation loss: %g' % log[-1]['val_loss'])
    print()
    print('Best training accuracy: {0:.3}'.format(best_training_accuracy))
    print('Final training accuracy: {0:.3}'.format(log[-1]['training_acc']))
    print('Best validation accuracy: {0:.3}'.format(best_validation_accuracy))
    print('Final validation accuracy: {0:.3}'.format(log[-1]['val_acc']))
    print()
    print('Best training F1: {0:.3}'.format(best_training_F1))
    print('Final training F1: {0:.3}'.format(log[-1]['training_F1']))
    print('Best validation F1:{0:.3}'.format( best_validation_F1))
    print('Final validation F1: {0:.3}'.format(log[-1]['val_F1']))
    print('Final validation precison: {0:.3}'.format(log[-1]['val_pre']))
    print('Final validation recall: {0:.3}'.format(log[-1]['val_recall']))

    print()
    print('Number of weights: %d' % len(weights))

    n_large_weights = sum(abs(w) > 0.01 for w in weights)
    n_large_weights = n_large_weights.item()
    _bias_as_float = bias.item()
    print('Bias: %g' % _bias_as_float)
    print('Number of weights with magnitude > 0.01: %d' % n_large_weights)

    _weights = [w.item() for w in weights]
    features = list(zip(_weights, vocabulary.keys()))
    features.sort()

    print()
    print('Top %d positive features:' % top_n)
    print('\n'.join('%g\t%s' % f for f in sorted(features[-top_n:], reverse=True)))
    print()
    print('Top %d negative features:' % top_n)
    print('\n'.join('%g\t%s' % f for f in features[:top_n]))


def display_log_record(iteration, log_record):
    print(('Epoch %d: ' % iteration) + ', '.join('%s %g' % (k, v) for k, v in log_record.items()))

def create_plots(title, log, weights, log_keys=None):
    _weights_to_list = [w.item() for w in weights]
    fig, (ax1, ax2) = pyplot.subplots(1, 2)
    fig.suptitle(title)
    plot_log(ax1, log, keys=log_keys)
    weight_histogram(ax2, _weights_to_list)
    pyplot.show()

def plot_log(ax, log, keys=None):
    if keys is None:
        keys = log[0].keys()

    max_loss = 0.0

    for key in keys:
        y = numpy.array([rec[key] for rec in log])
        ax.plot(y, label=key)
        max_loss = max(max_loss, y.max())

    ax.set_ylim(0.0, 1.1 * max_loss)
    ax.set_title('Learning curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

def weight_histogram(ax, weights):
    ax.set_title('Weight histogram')
    ax.set_xlabel('Count')
    ax.set_ylabel('Weight')
    ax.hist(weights, bins=30)


#Data loading
def load_awry_data(infile, task = 'comment', lowercase = False, remove_stopwords = False):
    comments = []
    labels = []

    with open(infile, 'r') as readin:
        wiki_json = json.load(readin)
        reduced_data = []
        i = 0
        for entry in wiki_json:
            tokens = entry['text'].split(' ')
            if lowercase:
                tokens = [t.lower() for t in tokens]
            if remove_stopwords:
                tokens = [t for t in tokens if t.lower() not in stopwords.words('english')]
            comments.append(tokens)
            if task == 'comment':
                attack = entry["awry_info"]["comment_has_personal_attack"]
            if task == 'conversation':
                attack = entry["awry_info"]["conversation_has_personal_attack"]
            labels.append(1 if attack == True else -1)

    #return comments,labels

    return SentimentData('Discussion comments', comments, labels)


class SentimentData:
    def __init__(self, name, sentences, labels):
        self.name = name
        self.sentences = sentences
        self.labels = labels

        unigrams = set()
        bigrams = set()
        for snt in sentences:
            unigrams.update(snt)
            bigrams.update(a + ' ' + b for a, b in zip(snt, snt[1:]))

        self.unigram_vocabulary = collections.OrderedDict((w, i) for i, w in enumerate(sorted(unigrams)))
        self.bigram_vocabulary = collections.OrderedDict((w, i) for i, w in enumerate(sorted(bigrams)))
        self.combined_vocabulary = \
            collections.OrderedDict((w, i) for i, w in enumerate(itertools.chain(sorted(unigrams), sorted(bigrams))))

        self.feature_type = 'unigram'
        self.vocabulary = self.unigram_vocabulary

    def __len__(self):
        return len(self.sentences)

    def select_feature_type(self, features):
        available_types = {
            'unigram': self.unigram_vocabulary,
            'bigram': self.bigram_vocabulary,
            'unigram+bigram': self.combined_vocabulary
        }
        self.feature_type = features
        self.vocabulary = available_types[features]

    def random_split(self, proportions):
        nexamples = len(self.sentences)
        sum_p = sum(proportions)
        idx = numpy.cumsum(numpy.array([0] + [int(p * nexamples / sum_p) for p in proportions], dtype=numpy.int32))
        perm = numpy.random.permutation(nexamples)
        return self._split_data(idx, perm)

    def train_val_test_split(self):
        idx = numpy.array([0, 4800, 5400, 6000], dtype=numpy.int32)
        perm = numpy.random.permutation(numpy.arange(6000))

        return self._split_data(perm, idx)

    def _split_data(self, perm, idx):
        split = []
        for i in range(len(idx) - 1):
            sub_sentences = [self.sentences[j] for j in perm[idx[i]:idx[i + 1]]]
            sub_labels = [self.labels[j] for j in perm[idx[i]:idx[i + 1]]]
            subset = SentimentData('%s_%d' % (self.name, i), sub_sentences, sub_labels)
            subset.unigram_vocabulary = self.unigram_vocabulary
            subset.bigram_vocabulary = self.bigram_vocabulary
            subset.combined_vocabulary = self.combined_vocabulary
            subset.select_feature_type(self.feature_type)
            split.append(subset)
        return split

    def features(self):
        for snt in self.sentences:
            if self.feature_type == 'unigram':
                unigrams = {self.unigram_vocabulary[w] for w in snt}
                yield unigrams
            elif self.feature_type == 'bigram':
                bigrams = {self.bigram_vocabulary[a + ' ' + b] for a, b in zip(snt, snt[1:])}
                yield bigrams
            elif self.feature_type == 'unigram+bigram':
                unigrams = {self.unigram_vocabulary[w] for w in snt}
                bigrams = {self.bigram_vocabulary[a + ' ' + b] for a, b in zip(snt, snt[1:])}
                yield unigrams | bigrams
            else:
                raise ValueError('Unknown feature type: ' + self.feature_type)





def sparse_to_dense(data, nfeatures):
    ds_features = []
    i = 0
    for ft in data.features():
        f_index = sorted(ft)
        ds = []
        for i in range(nfeatures):
            if len(f_index) > 0 and i == f_index[0]:
                ds.append(1)
                f_index.pop(0)
            else:
                ds.append(0)
        ds_features.append(ds)
    return ds_features
