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

    for sentence in sentences:
        words = sentence.split()
        if len(words) < 2:
            return False

    return True


def good_paragraph(paragraph):
    predicates = [ long_enough \
                 , good_start \
                 , good_end \
                 , good_characters \
                 # , no_caps \
                 , no_short_sentences ]

    return all((predicate(paragraph) for predicate in predicates))

def merge_and_filter_paragraphs(paragraphs):
    paragraphs = list(paragraphs)
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

def adjust_whitespace(paragraph):
    paragraph = paragraph.replace('\n', ' ')
    chars_with_padding = '.!?:,;"()'
    for c in chars_with_padding:
        paragraph = paragraph.replace(c, ' {} '.format(c))

    words = paragraph.split(' ')
    return ' '.join(filter(lambda str: str != '', words))

def sanitize_article(article_paragraphs):
    fixed_paragraphs = [adjust_whitespace(paragraph) for paragraph in article_paragraphs]
    return merge_and_filter_paragraphs(fixed_paragraphs)
