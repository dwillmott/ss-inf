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

def TransposeandConcat(x):
    xtransposed = k.permute_dimensions(x, (0,2,1,3))
    xconcat = k.concatenate([x, xtransposed], axis = -1)
    return xconcat

def TransposeandConcatShape(input_shape):
    shape = list(input_shape)
    shape[-1] = 2*shape[-1]
    return tuple(shape)
    

#def weighted_loss(y_true, y_pred):
    #y_true = k.
    #return