#!/usr/bin/env python3

from learn import train_nn_classifier, train_qda_classifier, train_lstsq_qda_classifier, \
                  train_gaussian_kde_classifier
from wordvec import WordEmbedding
from utils import lines_iter, to_ndarray, iterlen

from itertools import chain, islice, filterfalse
import matplotlib.pyplot as plt
from numpy import ones, zeros

def iter_ngrams(N, train_flag):
    file_name_pattern = 'data/{}-gram-train'
    if not train_flag:
        file_name_pattern = 'data/{}-gram-test'
    with open(file_name_pattern.format(N)) as f:
        lines = lines_iter(f)
        ngrams = map(lambda s: s.split(), lines)

        for ngram in ngrams:
            yield ngram

def embed_ngrams(ngrams, embedding):
    embed_ngram = lambda ngram: chain.from_iterable(map(embedding.__getitem__, ngram))
    return map(embed_ngram, ngrams)

def ngram_vec_iters(N, word, train_flag, embedding=None):
    assert N % 2 == 0

    if not embedding:
        embedding = WordEmbedding('data/word-embedding')
    
    nocomma = embed_ngrams(iter_ngrams(N, train_flag), embedding)

    middle_index = N // 2 # for a list of length N + 1
    def remove_middle(l):
        l.pop(middle_index)
        return l
    comma = iter_ngrams(N + 1, train_flag)
    comma = filter(lambda ngram: ngram[middle_index] == word, comma)
    comma = map(remove_middle, comma)
    comma = embed_ngrams(comma, embedding)

    return (comma, nocomma)

def visualize_classification(classify_fun, yes_samples, no_samples):
    yes_samples = list(map(classify_fun, yes_samples))
    no_samples = list(map(classify_fun, no_samples))

    correct_yes = iterlen(filter(None, yes_samples))
    incorrect_no = iterlen(filterfalse(None, yes_samples))
    incorrect_yes = iterlen(filter(None, no_samples))
    correct_no = iterlen(filterfalse(None, no_samples))

    correct = correct_yes + correct_no
    incorrect = incorrect_yes + incorrect_no

    correctness_rate = correct / (correct + incorrect)
    prior_yes = len(yes_samples) / (len(yes_samples) + len(no_samples))
    prior_correctness_rate = max(prior_yes, 1 - prior_yes)

    print('Correct Yes:', correct_yes)
    print('Incorrect False:', incorrect_no)
    print('Incorrect Yes:', incorrect_yes)
    print('Correct False:', correct_no)

    print('Correctness rate:', correctness_rate)
    print('Prior correctness rate:', prior_correctness_rate)

def main():
    embedding = WordEmbedding('data/word-embedding')

    (comma, nocomma) = ngram_vec_iters(4, ',', train_flag=True, embedding=embedding)

    comma = to_ndarray(comma)
    nocomma = to_ndarray(nocomma)
    comma_nocomma_ratio = len(comma) / len(nocomma)

    def train(trainer, sample_size):
        yes_samples = islice(comma, int(sample_size * comma_nocomma_ratio))
        no_samples = islice(nocomma, sample_size)

        return trainer(yes_samples, no_samples)

    nn_classify = train(train_nn_classifier, 1000)
    qda_classify = train(train_qda_classifier, 1000)
    lstsq_qda_classify = train(train_lstsq_qda_classifier, 10000)
    gaussian_kde_classify = train(train_gaussian_kde_classifier, 10000)

    (comma, nocomma) = ngram_vec_iters(4, ',', train_flag=False, embedding=embedding)
    comma = to_ndarray(comma)
    nocomma = to_ndarray(nocomma)
    comma_nocomma_ratio = len(comma) / len(nocomma)
    
    def test(classifier, sample_size):
        yes_samples = islice(comma, int(sample_size * comma_nocomma_ratio))
        no_samples = islice(nocomma, sample_size)

        visualize_classification(classifier, yes_samples, no_samples)
    
    print('nn:')
    test(nn_classify, 1000)
    print('qda:')
    test(qda_classify, 1000)
    print('lstsq_qda:')
    test(lstsq_qda_classify, 1000)
    print('gaussian_kde:')
    test(gaussian_kde_classify, 100)

if __name__ == '__main__':
    main()
