import keras
import numpy as np
import keras.backend as k

#from PIL import Image


def SelfCartesian(x):
    x_expanded = k.expand_dims(x, axis = -2)
    x_tiled = k.repeat_elements(x_expanded, k.int_shape(x_expanded)[1], axis=-2)
    x_transposed = k.permute_dimensions(x_tiled, (0,2,1,3))
    x_concat = k.concatenate([x_tiled, x_transposed], axis=-1)
    return x_concat


def SelfCartesianShape(input_shape):
    shape = list(input_shape)
    return [shape[0], shape[1], shape[1], shape[2]*2]


def weighted_cross_entropy(onehot_labels, logits):
    weight = 100
    class_weights = k.argmax(onehot_labels, axis = -1)
    class_weights = class_weights*weight + (1 - class_weights)
    unweighted_losses = k.categorical_crossentropy(onehot_labels, logits)
    weighted_losses = unweighted_losses * class_weights
    loss = k.mean(weighted_losses)
    return loss
