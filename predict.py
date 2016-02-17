#!/usr/bin/env python3

from learn import train_nn_classifier, train_qda_classifier, train_lstsq_qda_classifier, \
                  train_gaussian_kde_classifier, train_tophat_kde_classifier, \
                  train_random_forest_classifier, train_lstsq_random_forest_classifier
from wordvec import WordEmbedding
from utils import lines_iter, to_ndarray, genarr_first, duplicate_gen

from itertools import chain, islice
from functools import partial
import matplotlib.pyplot as plt
from numpy import ones, zeros
from time import time
from io import StringIO

def iter_ngrams(N, train_flag):
    file_name_pattern = 'data/{}-gram-train'
    if not train_flag:
        file_name_pattern = 'data/{}-gram-test'
    with open(file_name_pattern.format(N)) as f:
        lines = lines_iter(f)
        ngrams = map(lambda s: s.split(), lines)

        for ngram in ngrams:
            yield ngram

def embed_ngram(embedding, ngram):
    return to_ndarray(chain.from_iterable(map(embedding.__getitem__, ngram)))

def prepare_skipgrams(embedding, word, skipgrams):
    middle_index = lambda l: (len(l) - 1) // 2
    middle_is_word = lambda l: l[middle_index(l)] == word
    remove_middle = lambda l: l[ : middle_index(l)] + l[middle_index(l) + 1 : ]
    embed = lambda l: embed_ngram(embedding, l)
    
    skipgrams = filter(middle_is_word, skipgrams)
    skipgrams = map(remove_middle, skipgrams)
    skipgrams = map(embed, skipgrams)

    return skipgrams

def prepare_ngrams(embedding, ngrams):
    embed = lambda l: embed_ngram(embedding, l)
    return map(embed, ngrams)

def prepare_training_data(embedding, word, ngrams, skipgrams):
    ngrams = prepare_ngrams(embedding, ngrams)
    skipgrams = prepare_skipgrams(embedding, word, skipgrams)

    return (ngrams, skipgrams)

def prepare_test_data(embedding, word, ngrams, skipgrams):
    prep_ngrams = partial(prepare_ngrams, embedding)
    prep_skipgrams = partial(prepare_skipgrams, embedding, word)
    
    ngrams = duplicate_gen(ngrams)
    skipgrams = duplicate_gen(skipgrams)

    ngrams = genarr_first(prep_ngrams)(ngrams)
    skipgrams = genarr_first(prep_skipgrams)(skipgrams)

    return (ngrams, skipgrams)

class ClassifierAnalysis:
    def __init__(self, classifier, yes_test, no_test, yes_prior):
        self.yess = [(sample, classifier(vec)) for (vec, sample) in yes_test]
        self.nos = [(sample, classifier(vec)) for (vec, sample) in no_test]

        self.total_size = len(self.yess) + len(self.nos)
        self.yes_prior = yes_prior
        self.prior_correctness = max(self.yes_prior, 1 - self.yes_prior)

        self.yess_correct = [sample for (sample, result) in self.yess if result]
        self.yess_incorrect = [sample for (sample, result) in self.yess if not result]
        self.nos_correct = [sample for (sample, result) in self.nos if not result]
        self.nos_incorrect = [sample for (sample, result) in self.nos if not result]

        self.yes_correctness = len(self.yess_correct) / len(self.yess)
        self.no_correctness = len(self.nos_correct) / len(self.nos)

        self.algorithm_correctness = yes_prior * self.yes_correctness + (1 - yes_prior) * self.no_correctness

        self.error_rate_quotient = (1 - self.algorithm_correctness) / (1 - self.prior_correctness)

    def __str__(self):
        out = StringIO()
        p = partial(print, file=out)

        p('Prior correctness rate: {}'.format(self.prior_correctness))
        p('Algorithm correctness rate: {}'.format(self.algorithm_correctness))
        p('Error rate quotient algorithm/prior: {}'.format(self.error_rate_quotient))

        return out.getvalue()

def read_training_data(embedding, word, N, sample_number):
    ngrams = islice(iter_ngrams(N, train_flag=True), sample_number)
    n1grams = islice(iter_ngrams(N + 1, train_flag=True), sample_number)

    return prepare_training_data(embedding, word, ngrams, n1grams)

def read_test_data(embedding, word, N, sample_number):
    ngrams = islice(iter_ngrams(N, train_flag=True), sample_number)
    n1grams = islice(iter_ngrams(N + 1, train_flag=True), sample_number * 100)

    (ngrams, skipgrams) = prepare_test_data(embedding, word, ngrams, n1grams)

    ngrams = list(ngrams)
    skipgrams = list(skipgrams)
    yes_prior = len(skipgrams) / (sample_number + sample_number * 100)

    return (ngrams, skipgrams, yes_prior)

def analyze_nn(embedding):
    (ngrams, skipgrams) = read_training_data(embedding, ',', 4, 5000)

    classifier = train_nn_classifier(skipgrams, ngrams)

    (ngrams, skipgrams, yes_prior) = read_test_data(embedding, ',', 4, 1000)

    return ClassifierAnalysis(classifier, skipgrams, ngrams, yes_prior)

def analyze_qda(embedding):
    (ngrams, skipgrams) = read_training_data(embedding, ',', 4, 1000)

    classifier = train_qda_classifier(skipgrams, ngrams)

    (ngrams, skipgrams, yes_prior) = read_test_data(embedding, ',', 4, 1000)

    return ClassifierAnalysis(classifier, skipgrams, ngrams, yes_prior)

def analyze_lstsq_qda(embedding):
    (ngrams, skipgrams) = read_training_data(embedding, ',', 4, 10000)

    classifier = train_lstsq_qda_classifier(skipgrams, ngrams)

    (ngrams, skipgrams, yes_prior) = read_test_data(embedding, ',', 4, 1000)

    return ClassifierAnalysis(classifier, skipgrams, ngrams, yes_prior)

def analyze_gaussian_kde(embedding):
    (ngrams, skipgrams) = read_training_data(embedding, ',', 4, 1000)

    classifier = train_nn_classifier(skipgrams, ngrams)

    (ngrams, skipgrams, yes_prior) = read_test_data(embedding, ',', 4, 1000)

    return ClassifierAnalysis(classifier, skipgrams, ngrams, yes_prior)

def analyze_random_forest(embedding):
    (ngrams, skipgrams) = read_training_data(embedding, ',', 4, 100000)

    classifier = train_random_forest_classifier(skipgrams, ngrams)

    (ngrams, skipgrams, yes_prior) = read_test_data(embedding, ',', 4, 1000)

    return ClassifierAnalysis(classifier, skipgrams, ngrams, yes_prior)

def analyze_lstsq_random_forest(embedding):
    (ngrams, skipgrams) = read_training_data(embedding, ',', 4, 100000)

    classifier = train_lstsq_random_forest_classifier(skipgrams, ngrams)

    (ngrams, skipgrams, yes_prior) = read_test_data(embedding, ',', 4, 1000)

    return ClassifierAnalysis(classifier, skipgrams, ngrams, yes_prior)

def main():
    embedding100 = WordEmbedding('data/word-embedding-100')
    embedding10 = WordEmbedding('data/word-embedding-10')
    print(analyze_nn(embedding100))
    print(analyze_qda(embedding100))
    print(analyze_qda(embedding10))
    print(analyze_lstsq_qda(embedding100))
    print(analyze_gaussian_kde(embedding100))
    print(analyze_random_forest(embedding100))
    print(analyze_lstsq_random_forest(embedding100))

if __name__ == '__main__':
    main()
