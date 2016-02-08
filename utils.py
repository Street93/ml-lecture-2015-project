from time import sleep


def concat(generator):
    for sub_generator in generator:
        for value in sub_generator:
            yield value

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
