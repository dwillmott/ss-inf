import keras
import numpy as np
import keras.backend as k

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, Lambda, Reshape, Permute, Conv2D, Activation, Bidirectional, Reshape
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

#from PIL import Image


def get_activations(model, model_inputs, print_shape_only=True, layer_name=None):
    import keras.backend as K
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 1.]

    # Learning phase. 1 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations



np.set_printoptions(linewidth = 300, precision = 4, suppress = True)

def LambdaTile(x):
    xtiled = k.repeat_elements(x, length, axis=-1)
    return xtiled

def TileShape(input_shape):
    shape = list(input_shape)
    shape[-1] = shape[-1]**2
    return tuple(shape)

def TransposeandConcat(x):
    xtransposed = k.permute_dimensions(x, (0,2,1,3))
    xconcat = k.concatenate([x, xtransposed], axis = -1)
    return xconcat

def TransposeandConcatShape(input_shape):
    shape = list(input_shape)
    shape[-1] = 2*shape[-1]
    return tuple(shape)

#inputdata = np.load('data/zs-square.npy')
#print(inputdata.shape)

#inputdata = np.random.rand(1, 5, 7)
#inputdata = inputdata.reshape((1, 5, 7))
#inputdata = 

length = 135
h1size = 50

inputdata = np.load('data/crw5s.npy')[:,:length,1:7]
print(inputdata.shape)

target = np.load('data/5s-square.npy')[:,:length,:length,1:]
print(target.shape)

print(np.count_nonzero(target[0,:,:,0]))

samples = 4

inputdata = inputdata[0:samples]
target = target[0:samples]

print(target[0, 40:55, 25:35, 0])
print(target[0, 25:35, 40:55, 0])


model = Sequential()
model.add(Bidirectional(LSTM(h1size, return_sequences = True), input_shape = (length,6)))
model.add(Permute((2,1)))
model.add(Lambda(LambdaTile, output_shape = TileShape))
model.add(Reshape((h1size*2,length,length)))
model.add(Permute((3,2,1)))
model.add(Lambda(TransposeandConcat, output_shape = TransposeandConcatShape))
#model.add(Conv2D(filters=50, kernel_size=11, activation='relu', padding='same'))
#d = Conv2D(filters=10, kernel_size=5, activation='relu', padding='same')
model.add(Conv2D(filters=20, kernel_size=5, activation='relu', padding='same'))
model.add(Conv2D(filters=2, kernel_size=5, activation='relu', padding='same'))
model.add(Reshape((length**2, 2)))
model.add(Activation('softmax'))


opt = RMSprop(lr=0.0001)
#opt = Adam(lr = 0.0000001)
model.compile(optimizer=opt,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'],
              sample_weight_mode = "temporal")

#print(inputdata[0])

out = model.predict(inputdata)

print(inputdata.shape)
print(target.shape)
print(out.shape)
#print(out)

print(model.summary())

#activations = get_activations(model, inputdata)
#print(activations[0])
#print(activations[1])

#sampleweights = target

print(np.count_nonzero(target[:,:,:,1]))

sample_weight = (target*10)[:,:,:,0].reshape((samples, length**2))
sample_weight = sample_weight + np.ones(sample_weight.shape)*0.001

for j in range(samples):
    print(sample_weight.reshape(samples, 135, 135, 1)[j, 40:55, 25:35, 0])

#sample_weight = np.ones((2,135**2))

for i in range(200):
    print(i)
    model.fit(inputdata, target.reshape((samples, length**2, 2)), epochs = 20, sample_weight = sample_weight)
    
    print(model.evaluate(inputdata, target.reshape((samples, length**2, 2))))
    pred = model.predict(inputdata)
    pred = pred.reshape(samples, 135, 135, 2)
    
    for j in range(samples):
        print(pred[j, 40:55, 25:35, 0])
        thres = np.zeros(pred.shape)
    
        #thres[pred > 0.5] = 1
        ##print(thres[0, 40:55, 25:35, 0])
        #accura = thres[0,:,:,0] - target[0,:,:,0]
        #print('incorrect predictions: ', np.sum(np.abs(accura)))
    
    #thres = np.zeros(pred.shape)
    
    #thres[pred > 0.5] = 1
    ##print(thres[0, 40:55, 25:35, 0])
    #accura = thres[0,:,:,0] - target[0,:,:,0]
    #print('incorrect predictions: ', np.sum(np.abs(accura)))
    #print()
    #im = Image.fromarray((pred[0,:,:,0]*255).astype()
    #im.save('outputpicture.png')
    #activations = get_activations(model, inputdata)
    #print(activations[0])
    

