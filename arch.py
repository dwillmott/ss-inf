import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, LSTM, Lambda, Conv2D, Conv2DTranspose, Activation, Bidirectional, Concatenate, BatchNormalization, TimeDistributed
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras.models import Model


# CUSTOM KERAS LAYERS

def SelfCartesian(x):
    newshape = K.stack([1, 1, K.shape(x)[1], 1])
    
    x_expanded = K.expand_dims(x, axis = -2)
    x_tiled = K.tile(x_expanded, newshape)
    x_transposed = K.permute_dimensions(x_tiled, (0,2,1,3))
    x_concat = K.concatenate([x_tiled, x_transposed], axis=-1)
    return x_concat


def SelfCartesianShape(input_shape):
    shape = list(input_shape)
    return [shape[0], shape[1], shape[1], shape[2]*2]


# ARCHITECTURE

def makemodel(LSTMlayers, BN, weight, reg, lr):
    
    l2reg = l2(reg)
    weight = K.constant(weight)
    
    def weighted_binary_cross_entropy(labels, logits):
        class_weights = labels*weight + (1 - labels)
        unweighted_losses = K.binary_crossentropy(target=labels, output=logits)
        
        weighted_losses = unweighted_losses * class_weights
        
        loss = K.mean(tf.matrix_band_part(K.squeeze(weighted_losses, -1), 0, -1))
        return loss
    
    
    inputs = Input(shape=(None, 5))
    
    if LSTMlayers:
        h1_lstm = Bidirectional(LSTM(75, return_sequences = True))(inputs)
        if LSTMlayers > 1:
            h1_lstm = Bidirectional(LSTM(50, return_sequences = True))(h1_lstm)
        h1_lstmout = TimeDistributed(Dense(20))(h1_lstm)
        h1 = Concatenate(axis=-1)([inputs, h1_lstmout])
        h1square = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(h1)
    else:
        h1square = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)

    h2square_1 = Conv2D(filters=20, kernel_size=13, use_bias=False, kernel_regularizer = l2reg, padding='same')(h1square)
    h2square_2 = Conv2D(filters=20, kernel_size=9, use_bias=False, kernel_regularizer = l2reg, padding='same')(h1square)
    h2square_3 = Conv2D(filters=20, kernel_size=5, use_bias=False, kernel_regularizer = l2reg, padding='same')(h1square)
    h2square_a = Concatenate(axis=-1)([h2square_1, h2square_2, h2square_3])
    if BN:
        h2square_b = BatchNormalization(axis=-1)(h2square_a)
        h2square = Activation('relu')(h2square_b)
    else:
        h2square = Activation('relu')(h2square_a)

    h3square_1 = Conv2D(filters=20, kernel_size=9, use_bias=False, kernel_regularizer = l2reg, padding='same')(h2square)
    h3square_2 = Conv2D(filters=20, kernel_size=5, use_bias=False, kernel_regularizer = l2reg, padding='same')(h2square)
    h3square_a = Concatenate(axis=-1)([h3square_1, h3square_2])
    if BN:
        h3square_b = BatchNormalization(axis=-1)(h3square_a)
        h3square = Activation('relu')(h3square_b)
    else:
        h3square = Activation('relu')(h3square_a)

    h4square_1 = Conv2D(filters=20, kernel_size=5, activation='relu', kernel_regularizer = l2reg, padding='same')(h3square)
    sequencesquare = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)
    h4square = Concatenate(axis=-1)([h4square_1, sequencesquare])

    output = Conv2D(filters=1, kernel_size=3, activation='sigmoid', kernel_regularizer = l2reg, padding='same')(h4square)

    #opt = Adam(lr=lr)
    model = Model(input = inputs, output = output)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss = weighted_binary_cross_entropy)
    
    return model, opt