from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle

from .custom_layers import Avg2MaxPooling, SEBlock, DepthwiseSeparableConv

print(tf.__version__)
print(np.__version__)

model = load_model("model/apple_model.keras", custom_objects = {
    "Avg2MaxPooling": Avg2MaxPooling,
    "SEBlock": SEBlock,
    "DepthwiseSeparableConv": DepthwiseSeparableConv
})

with open('model/encode.pkl', mode = 'rb') as e:
    encode = pickle.load(e)



