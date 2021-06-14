import keras
from keras.models import load_model

__all__ = ['bagnet9', 'bagnet17', 'bagnet33']

model_urls = {
    'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet8.h5',
    'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet16.h5',
    'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet32.h5'
}


def bagnet9():
    model_path = keras.utils.get_file(
        'bagnet8.h5',
        model_urls['bagnet9'],
        cache_subdir='models',
        file_hash='5b70adc7c4ff77d932dbba485a5ea1d333a65e777a45511010f22e304a2fdd69')

    return load_model(model_path)


def bagnet17():
    model_path = keras.utils.get_file(
        'bagnet16.h5',
        model_urls['bagnet17'],
        cache_subdir='models',
        file_hash='b262dfee15a86c91e6aa21bfd86505ecd20a539f7f7c72439d5b1d352dd98a1d')

    return load_model(model_path)


def bagnet33():
    model_path = keras.utils.get_file(
        'bagnet32.h5',
        model_urls['bagnet33'],
        cache_subdir='models',
        file_hash='96d8842eec8b8ce5b3bc6a5f4ff3c8c0278df3722c12bc84408e1487811f8f0f')

    return load_model(model_path)
