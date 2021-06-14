from .tf_hub_model_url import tfhub_urls
import tensorflow_hub as hub
import tensorflow as tf


def build_model_from_hub(model_name):

    model_url = tfhub_urls.get(model_name)
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url)
    ])
        
    return model

