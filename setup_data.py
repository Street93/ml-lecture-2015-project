#!/usr/bin/env python3

import spiegel
from spiegel import load_raw_articles, SpiegelIssue, IssueDoesNotExist
from sanitize import sanitize_article
from utils import retrying
import texttools
from texttools import create_word_embedding, WordEmbedding, random_corpus_ngrams
from config import dataconfigs, runconfigs

import gzip
from numpy import random
from os import makedirs, path
import requests
from itertools import islice
from pathlib import Path
from collections import namedtuple
from io import StringIO

def download_spiegel(first_issue=SpiegelIssue(1970, 1)):
    makedirs('data/spiegel', exist_ok=True)
    issues = (SpiegelIssue(year, week) for year in range(first_issue.year, 2010) \
                                       for week in range(first_issue.week, 53))

    for issue in issues:
        @retrying( requests.exceptions.ConnectionError \
                 , retries=1 \
                 , retry_delay=120 )
        def retrying_load():
            return load_raw_articles(issue, topath='data/spiegel')

        try:
            retrying_load()
        except IssueDoesNotExist:
            pass
        except FileExistsError:
            pass
        except requests.exceptions.ConnectionError as exc:
            print('Failed to download issue {}-{:02}'.format(year, week), file=stderr)
            print(exc, file=stderr)

def create_embedding(dataconfig):
    makedirs('data/embeddings', exist_ok=True)
    corpuspath = dataconfig.corpuspath()

    embeddingpath = dataconfig.embeddingpath()
    if path.isfile(embeddingpath):
        return

    create_word_embedding( infile=str(corpuspath) \
                         , outfile=str(embeddingpath) \
                         , size=dataconfig.dimension \
                         , estimator=dataconfig.estimator \
                         , negative=dataconfig.negative \
                         , downsample=dataconfig.downsampling )
    
def create_vocabulary(dataconfig):
    makedirs('data/vocabularies', exist_ok=True)
    corpuspath = dataconfig.corpuspath()

    vocabularypath = dataconfig.vocabularypath()
    if path.isfile(vocabularypath):
        return

    texttools.create_vocabulary(corpuspath, vocabularypath)

def create_ngrams(dataconfig):
    makedirs('data/ngrams', exist_ok=True)
    ns = [dataconfig.ngram_size, dataconfig.ngram_size + 1]
    vocab = None
    corpuspath = dataconfig.corpuspath()
    vocabularypath = dataconfig.vocabularypath()
    for n, ngrampath in zip(ns, dataconfig.ngrampaths()):
        if path.exists(ngrampath):
            continue

        if vocab is None:
            vocab = texttools.load_vocabulary(vocabularypath)

        def valid_ngram(ngram):
            return all(map(lambda word: word in vocab, ngram))

        ngrams = random_corpus_ngrams(corpuspath, n, predicate=valid_ngram)
        
        with gzip.open(ngrampath, mode='wt') as f:
            for ngram in ngrams:
                print(' '.join(ngram), file=f)


def create_corpora():
    makedirs('data/corpora', exist_ok=True)
    with gzip.open('data/corpora/spiegel-full.gz', mode='wt') as f:
        articles = map(sanitize_article, spiegel.iter_issues_articles())
        for article in articles:
            for paragraph in article:
                print(paragraph, file=f)

def download_data():
    print('Downloading Spiegel issues... ')
    download_spiegel()
    print('Done.')

def setup_corpora():
    print('Scraping text corpora... ')
    create_corpora()
    print('Done.')

def setup_data(dataconfigs):
    print('Creating text embeddings... ')
    for dataconfig in dataconfigs:
        create_embedding(dataconfig)
    print('Done.')
    print('Creating vocabularies... ')
    for dataconfig in dataconfigs:
        create_vocabulary(dataconfig)
    print('Done.')
    print('Extracting ngrams... ')
    for dataconfig in dataconfigs:
        create_ngrams(dataconfig)
    print('Done.')

def main():
    download_data()
    setup_corpora()
    setup_data(dataconfigs)

if __name__ == '__main__':
    main()
