from __future__ import print_function
from __future__ import unicode_literals
import argparse
import csv
import keras
import pandas as pd
import io
from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, InputLayer
import numpy as np
import librosa
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten


def build_model():
    """
    Builds up the SoundNet model and loads the weights from a given model file (8-layer model is stored at models/sound8.npy).
    :return: The model built according to architecture and weights pre-stablished
    """
    model_weights = np.load('/Users/christos/PycharmProjects/pythonProject/sound8.npy', allow_pickle=True,
                            encoding='latin1').item()
    model = Sequential()
    # Input layer: audio raw waveform (1,length_audio,1)
    model.add(InputLayer(batch_input_shape=(1, None, 1)))

    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},  # pool1

                         {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                          'kernel_size': 32, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},  # pool2

                         {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                          'kernel_size': 16, 'conv_strides': 2},

                         {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                          'kernel_size': 8, 'conv_strides': 2},

                         {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2,
                          'pool_size': 4, 'pool_strides': 4},  # pool5

                         {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                          'kernel_size': 8, 'conv_strides': 2},  # output: VGG 401 classes


                         ]

    for x in filter_parameters:
        # for each [zero_padding - conv - batchNormalization - relu]
        model.add(ZeroPadding1D(padding=x['padding']))
        model.add(Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         padding='valid'))
        weights = model_weights[x['name']]['weights'].reshape(model.layers[-1].get_weights()[0].shape)
        biases = model_weights[x['name']]['biases']

        model.layers[-1].set_weights([weights, biases])  # set weights in convolutional layer

        if 'conv8' not in x['name']:
            gamma = model_weights[x['name']]['gamma']
            beta = model_weights[x['name']]['beta']
            mean = model_weights[x['name']]['mean']
            var = model_weights[x['name']]['var']

            model.add(BatchNormalization())
            model.layers[-1].set_weights([gamma, beta, mean, var])  # set weights in batchNormalization
            model.add(Activation('relu'))

        if 'pool_size' in x:
            # add 3 pooling layers
            model.add(MaxPooling1D(pool_size=x['pool_size'],
                                   strides=x['pool_strides'],
                                   padding='valid'))

    return model




def preprocess(audio):
    audio *= 256.0  # SoundNet requires an input range between -256 and 256
    # reshaping the audio data, in this way it fits into the graph (batch_size, num_samples, num_filter_channels)
    audio = np.reshape(audio, (1, -1))
    return audio


def load_audio(audio_file):
    sample_rate = 22050  # SoundNet works on monophonic-audio files with sample rate of 22050.
    audio, sr = librosa.load(audio_file, dtype='float32', sr=22050, mono=True) #load audio
    audio = preprocess(audio) # pre-process using SoundNet parameters
    return audio






f = open('/Users/christos/PycharmProjects/pythonProject/SoundNet-keras/soundnet_keras-master/categories/categories_places2.txt', 'r')
categories = f.read().split('\n')


ff = open('/Users/christos/PycharmProjects/pythonProject/SoundNet-keras/soundnet_keras-master/categories/categories_imagenet.txt', 'r')
image_objectts = ff.read().split('\n')

def predictions_to_scenes(prediction):
    scenes = []
    for p in range(prediction.shape[1]):
        scenes.append(categories[np.argmax(prediction[0, p, :])])
    return scenes



def predictions_to_objects(predictionsc):
    objects = []
    for p in range(predictionsc.shape[1]):
        objects.append(image_objectts [np.argmax(predictionsc[0, p, :])])
    return objects


model = build_model()
# model.summary()

# Parsing the files


parser = argparse.ArgumentParser()
parser.add_argument("csv_file_path")
parser.add_argument("--encoding", default="utf_8")

parser.add_argument('Transfer_learning_layer', metavar='Transfer_learning_layer', type=int,
                    help='specify the layer to extract features for transfer learning')

parser.add_argument('Epochs', metavar='Epochs', type=int,
                    help='specify the number of epochs')

parser.add_argument('Batch_size', metavar='Batch_size', type=int,
                    help='specify the batch size')

parser.add_argument('Dataset_size', metavar='Dataset_size', type=int,
                    help='specify the Dataset_size 10,40,50')

parser.add_argument('Learning_Rate_value', metavar='Learning_Rate_value', type=float,
                    help='specify the Learning_Rate_value')

parser.add_argument('Save_name', metavar='Save_name', type=str,
                    help='specify the name of the .h5 file')


args= parser.parse_args()

csv_reader = csv.reader(
    io.open(args.csv_file_path, "r", encoding=args.encoding),
    delimiter=",",
    quotechar='"'
)



testt = args.csv_file_path
Input_epochs = args.Epochs
Input_batchsize = args.Batch_size
Input_feature_layer = args.Transfer_learning_layer
dataset_size = args.Dataset_size
lr = args.Learning_Rate_value
save_name = args.Save_name




test = pd.read_csv(testt,sep=',')


if dataset_size == 10:
    esc10 = test[test['esc10'] == True]   #### dokimazw gia 40 classes
elif dataset_size == 40:
    esc10 = test[test['esc10'] == False]
else:
    esc10 = test  ###full 50 classes





data = [] #load the audiofiles
data2 = []
list_target = esc10['target'] #list of targets in dataset ESC10
list_category = esc10['category'] #list of categories (scenes) in dataset ESC10




#loop to load audiofiles
for file_name in list(esc10['filename']):
    audio = load_audio('/Users/christos/PycharmProjects/pythonProject/SoundNet-keras/ESC-50-master/audio/'+ file_name)
    data.append(audio)
    print(len(data))


def getActivations(data, number_layer):
    intermediate_tensor = []
    # get Hidden Representation function
    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[number_layer].output])
    # just for information
    ex = get_layer_output([data[155]])[0]
    #print('Dimension layer {}: {}'.format(number_layer, ex.shape))

    for audio in data:
        # get Hidden Representation
        layer_output = get_layer_output([audio])[0]  # multidimensional vector
        #print(layer_output.shape,'layer output')
        tensor = layer_output.reshape(1, -1)  # change vector shape to 1 (tensor)
        #print(tensor.shape,'tensor')
        intermediate_tensor.append(tensor[0])  # list of tensor activations for each object in Esc10

    return intermediate_tensor






#activations22 = getActivations(data,int(Input_feature_layer)) #get activation tensor for the 22nd layer in the model (pool5)
# print(activations22)

layers = [0,4,9,15,22,28] # List of important hidden layers

#obtain vector layer 22 INPUT
x = np.asarray(getActivations(data,22)) #Transfer learning
#print('Dataset input shape', x.shape)
y = np.asarray(list_target)


toClass = {} #dictionary to translate values of classes
toCategory = {} #dictionary with new values of categories
names = np.asarray(list_category) #array with names of categories
i = 0
# loop to change all target from Esc50 values to 0-9 range
for position,target in enumerate(y):
    if target not in toClass:
        toClass[target] = i #dictionary classes
        toCategory[i] = names[position] #dictionary categories
        i += 1
#
# print("Dictionary of classes: ",toClass)
# print("Values representation: ",toCategory)
# print(names)


#change vector of classes
Y = []
for i in y:
    Y.append(toClass[i])
Y = np.asarray(Y)




num_classes = dataset_size
print(num_classes)



#Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state = 25)
z_train = to_categorical(y_train, num_classes) # convert class vectors to binary class matrices
z_test = to_categorical(y_test, num_classes)






classifier = Sequential()
classifier.add(Dense(num_classes, activation='softmax',input_shape=(3328,)))
classifier.summary()

from keras.optimizers import Adam
classifier.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr),metrics=['accuracy'])

batch_size =Input_batchsize  ###64
epochs = Input_epochs   #100


hist_classifier = classifier.fit(x_train,z_train, validation_data=(x_test,z_test), epochs=epochs,batch_size=batch_size)

classifier.save(save_name,save_format='h5')
