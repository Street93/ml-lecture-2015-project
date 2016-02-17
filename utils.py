from time import sleep
from numpy import ndarray, array, float32, float64


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

def to_ndarray(it, dtype=float32):
    if type(it) in [float, int, bool, ndarray, float32, float64]:
        return it

    return array([to_ndarray(subgen) for subgen in it], dtype=dtype)
    

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

class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

def genarr_first(genarr):
    def newarr(gen):
        current_snd = None
        def set_and_discard_snd(tup):
            nonlocal current_snd
            current_snd = tup[1]
            return tup[0]
        
        gen = map(set_and_discard_snd, gen)

        for transformed_fst in genarr(gen):
            yield (transformed_fst, current_snd) 

    return newarr

def duplicate_gen(gen):
    for x in gen:
        yield (x, x)
