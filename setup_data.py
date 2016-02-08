#!/usr/bin/env python3

from spiegel import load_raw_articles, SpiegelIssue, IssueDoesNotExist
from utils import retrying
from os import makedirs
import requests

def download_spiegel(first_issue=SpiegelIssue(1990, 1)):
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
        except requests.exceptions.ConnectionError as exc:
            print('Failed to download issue {}-{:02}'.format(year, week), file=stderr)
            print(exc, file=stderr)

def main():
    download_spiegel()

if __name__ == '__main__':
    main()
