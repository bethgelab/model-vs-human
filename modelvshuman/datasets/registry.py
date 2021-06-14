from collections import defaultdict

_dataset_registry = {}  # mapping of dataset names to entrypoint fns


def register_dataset(name):
    def inner_decorator(fn):
        # add entries to registry dict/sets
        model_name = fn.__name__
        _dataset_registry[name] = model_name
        return fn
    return inner_decorator


def list_datasets():
    """ Return list of available dataset names, sorted alphabetically
    """
    return _dataset_registry
