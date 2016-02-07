#!/usr/bin/env python3

from lxml import html
import requests
from collections import namedtuple
from itertools import chain
from time import sleep
from sys import stderr

SpiegelIssue = namedtuple('SpiegelIssue', 'year week')

class IssueDoesNotExist(Exception):
    pass

def load_index(spiegel_issue):
    (year, week) = spiegel_issue
    indexUrl = 'http://www.spiegel.de/spiegel/print/index-{}-{}.html'.format(year, week)
    page = requests.get(indexUrl)
    if page.status_code != requests.codes.ok:
        raise IssueDoesNotExist()
    doc = html.fromstring(page.content)

    linkElements = doc.cssselect('.spiegel-contents > ul a')

    links = []
    for linkElement in linkElements:
        links.append('https://www.spiegel.de' + linkElement.get('href'))

    return links

def load_raw_article_paragraphs(url):
    page = requests.get(url)
    doc = html.fromstring(page.content)

    paragraphElements = doc.cssselect('.artikel > p')

    paragraphs = []
    for paragraphElement in paragraphElements:
        if paragraphElement.text:
            paragraphs.append(paragraphElement.text)

    return paragraphs

def long_enough(paragraph):
    return len(paragraph) >= 10

def good_start(paragraph):
    return paragraph[0].isalpha() and paragraph[0].isupper()

def good_end(paragraph):
    good_ends = ['.', '?', '!', '."', '?"', '!"']

    return any((paragraph.endswith(end) for end in good_ends))

def good_characters(paragraph):
    good_chars = 'abcdefghijklmnopqrstuvwxyz' \
               + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' \
               + 'äöüÄÖÜß' \
               + '0123456789' \
               + '.!?:,;"-()' \
               + ' '

    return all((c in good_chars for c in paragraph))

def no_caps(paragraph):
    i = 0
    while i <= len(paragraph) - 2:
        s = paragraph[i : i + 2]
        if s.isalpha() and s.isupper():
            return False
        i += 1

    return True

def split_at(string, predicate):
    substr_start = 0
    i = 0
    while i != len(string):
        if(predicate(string[i])):
            yield string[substr_start : i]
            substr_start = i + 1
        i += 1 

    yield string[substr_start : ]

def no_short_sentences(paragraph):
    sentence_end_chars = ['.', '!', '?', ':']
    sentences = list(split_at(paragraph, lambda c: c in sentence_end_chars))[ : -1]
    # print(sentences)

    for sentence in sentences:
        words = sentence.split()
        # print(words)
        if len(words) < 2:
            return False

    return True


def good_paragraph(paragraph):
    predicates = [ long_enough \
                 , good_start \
                 , good_end \
                 , good_characters \
                 , no_caps \
                 , no_short_sentences ]

    return all((predicate(paragraph) for predicate in predicates))

def merge_and_filter_paragraphs(paragraphs):
    result = []

    i = 0
    while i != len(paragraphs):
        good_paragraphs = []
        while i != len(paragraphs) and good_paragraph(paragraphs[i]):
            good_paragraphs.append(paragraphs[i])
            i += 1

        if good_paragraphs != []:
            result.append(' '.join(good_paragraphs))

        bad_paragraphs = []
        while i != len(paragraphs) and not good_paragraph(paragraphs[i]):
            bad_paragraphs.append(paragraphs[i])
            i += 1

        if bad_paragraphs != [] and \
                not good_end(bad_paragraphs[-1]) and \
                i != len(paragraphs):
            i += 1

    return result

def fix_whitespace(paragraph):
    paragraph = paragraph.replace('\n', ' ')

    words = paragraph.split(' ')
    return ' '.join(filter(lambda str: str, words))

def sanitize_paragraphs(article_paragraphs):
    fixed_paragraphs = [fix_whitespace(paragraph) for paragraph in article_paragraphs]
    return merge_and_filter_paragraphs(fixed_paragraphs)

def load_and_sanitize(article_url):
    return sanitize_paragraphs(load_raw_article_paragraphs(article_url))

def load_issue_paragraphs(spiegel_issue, mapfun=map):
    article_urls = load_index(spiegel_issue)

    articles = mapfun(load_and_sanitize, article_urls)
    return chain(*articles)

def make_retrying(fun, exception_class=None, max_retries=1, wait_duration=None):
    def newfunc():
        retries = 0
        while True:
            try:
                return fun()
            except exception_class as exc:
                if retries == max_retries:
                    raise exc
                else:
                    retries += 1

                if wait_duration is not None:
                    sleep(wait_duration)

    return newfunc
            

def save_issue(issue):
    (year, week) = issue

    def load():
        return load_issue_paragraphs(issue)

    retrying_load = \
        make_retrying( load \
                     , exception_class=requests.exceptions.ConnectionError \
                     , max_retries=1 \
                     , wait_duration=180) 

    paragraphs = retrying_load()

    with open('spiegel-{}-{:02}'.format(year, week), 'w') as f:
        for paragraph in paragraphs:
            print(paragraph, file=f)
    

def main():
    issues = (SpiegelIssue(year, week) for year in range(1990, 2010)
                                       for week in range(1, 53))
    for issue in issues:
        try:
            save_issue(issue)
        except IssueDoesNotExist:
            pass
        except requests.exceptions.ConnectionError as exc:
            print('Failed to download issue {}-{:02}'.format(year, week), file=stderr)
            print(exc, file=stderr)


if __name__ == '__main__':
    main()
