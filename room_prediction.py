import glob
import os

import numpy as np
import requests
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

physical_devices = tf.config.list_physical_devices('CPU')
# To find out which devices your operations and tensors are assigned to
tf.config.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(False)
# NOTE: uncomment this if train using GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
image_shape = 240
model_path = "Room_prediction_model"
CATEGORIES = ['bathroom', 'bedroom', 'dining room', 'exterior', 'interior', 'kitchen', 'living room']

class Model():
    def __init__(self,model_path ):
        self.model = load_model(model_path)
    def predict(self,X):
        return [CATEGORIES[i.argmax()]for i in self.model.predict(X)]