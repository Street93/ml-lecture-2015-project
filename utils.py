from time import sleep


def lines_iter(f):
    for line in f:
        yield line

def iterlen(iterator):
    return sum(1 for _ in iterator)

def retrying(exception_class, retries=1, retry_delay=None):
    def wrap(func):
        def newfunc(*args, **kwargs):
            retry_num = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exception_class as exc:
                    if retry_num == retries:
                        raise exc
                    else:
                        retry_num += 1

                    if retry_delay:
                        sleep(retry_delay)
        return newfunc

    return wrap

def subsequences(iterable, length):
    iterator = iter(iterable)
    current_subseq = []
    try:
        for i in range(0, length):
            current_subseq.append(next(iterator))
    except StopIteration:
        return

    yield iter(current_subseq)

    for item in iterator:
        current_subseq.pop(0)
        current_subseq.append(item)

        yield iter(current_subseq)
