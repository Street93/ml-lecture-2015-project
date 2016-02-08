#!/usr/bin/env python3

from lxml import html
import requests
from collections import namedtuple
from itertools import chain
from time import sleep
from sys import stderr
from pathlib import Path
import tarfile
from io import BytesIO

class SpiegelIssue(namedtuple('SpiegelIssue', 'year week')):
    def __str__(self):
        return 'spiegel-{}-{:02}'.format(self.year, self.week)

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

def load_raw_articles(issue, topath='.'):
    article_urls = load_index(issue)

    archive_path = Path(topath) / (str(issue) + '.tar.gz')
    with tarfile.open(str(archive_path), mode='x:gz') as tar:
        for url in article_urls:
            response = requests.get(url)
            if response.ok:
                file_name = url.split('/')[-1]
                tarinfo = tarfile.TarInfo(file_name)
                tarinfo.size = len(response.content)
                tar.addfile(tarinfo, BytesIO(response.content))

def scrape_paragraphs(page_source_bytes):
    doc = html.fromstring(page_source_bytes)

    paragraphElements = doc.cssselect('.artikel > p')

    paragraphs = []
    for paragraphElement in paragraphElements:
        if paragraphElement.text:
            paragraphs.append(paragraphElement.text)

    return paragraphs

def iter_articles(issue_archive_path):
    with tarfile.open(issue_archive_path) as tar:
        for member in tar:
            f = tar.extractfile(member)
            yield scrape_paragraphs(f.read())

def iter_issues_articles(issues=None, data_path='data/spiegel'):
    def iter_paths():
        if issues == None:
            for path in Path(data_path).iterdir():
                yield path
        else:
            for issue in issues:
                yield Path(data_path) / (str(issue) + '.tar.gz')
    
    for issue_path in iter_paths():
        for article in iter_articles(str(issue_path)):
            yield article
