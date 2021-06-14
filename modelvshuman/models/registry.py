from collections import defaultdict

__all__ = ['list_models']

_model_registry = defaultdict(list)  # mapping of model names to entrypoint fns


def register_model(framework):
    def inner_decorator(fn):
        # add entries to registry dict/sets
        model_name = fn.__name__
        _model_registry[framework].append(model_name)
        return fn
    return inner_decorator


def list_models(framework):
    """ Return list of available model names, sorted alphabetically
    """
    return list(_model_registry[framework])

