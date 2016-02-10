#!/usr/bin/env python3

import spiegel
from spiegel import load_raw_articles, SpiegelIssue, IssueDoesNotExist
from sanitize import sanitize_article
from utils import retrying
from wordvec import create_word_embedding, WordEmbedding
from ngram import random_corpus_ngrams

from numpy import random
from os import makedirs
import requests
from itertools import islice

def download_spiegel(first_issue=SpiegelIssue(1970, 1)):
    makedirs('data/spiegel', exist_ok=True)
    issues = (SpiegelIssue(year, week) for year in range(first_issue.year, 2010) \
                                       for week in range(first_issue.week, 53))

    for issue in issues:
        @retrying( requests.exceptions.ConnectionError \
                 , retries=1 \
                 , retry_delay=120)
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

def create_corpus():
    with open('data/text-corpus', mode='w') as f:
        articles = map(sanitize_article, spiegel.iter_issues_articles())
        for article in articles:
            for paragraph in article:
                print(paragraph, file=f)

def create_embedding():
    create_word_embedding('data/text-corpus', 'data/word-embedding')

def create_ngrams():
    random.seed(4)
    embedding = WordEmbedding('data/word-embedding')
    def valid_ngram(ngram):
        def known_word(word):
            try:
                embedding[word]
                return True
            except:
                return False
        return all(map(known_word, ngram))

    for N in [4, 5, 10, 11, 20, 21]:
        ngrams = random_corpus_ngrams( 'data/text-corpus' \
                                    , N \
                                    , number=1000000 \
                                    , predicate=valid_ngram )

        with open('data/{}-gram-train'.format(N), mode='w') as f:
            for ngram in islice(ngrams, 900000):
                print(' '.join(ngram), file=f)
        with open('data/{}-gram-test'.format(N), mode='w') as f:
            for ngram in ngrams:
                print(' '.join(ngram), file=f)

def main():
    download_spiegel()
    create_corpus()
    create_embedding()
    create_ngrams()

if __name__ == '__main__':
    main()
