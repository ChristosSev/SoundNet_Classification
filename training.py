from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, InputLayer
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import librosa
import sklearn

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

from keras.utils import plot_model

#Review of the model and architecture parameters
model = build_model()
# model.summary()


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


# let's load an audio file
audiot,sr = librosa.load('/Users/christos/PycharmProjects/pythonProject/SoundNet-keras/soundnet_keras-master/railroad_audio.wav',
                         dtype='float32', sr=22050, mono=True) #load audio




def predict_scene_from_audio_file(audio_file):
    model = build_model()
    audio = load_audio(audio_file)
    return model.predict(audio)

# Use previous function to predict and show scenes in the audio
prediction = predict_scene_from_audio_file('/Users/christos/PycharmProjects/pythonProject/SoundNet-keras/soundnet_keras-master/railroad_audio.wav')
#print("output model shape", prediction.shape)


def findMax(vector): #find 3 classes max
    maximus = []
    for i in range(3):
        index = np.argmax(vector)
        f = [index, max(vector)]
        vector[index] = 0
        maximus.append(f)
    return maximus

def putText(datos,x):
    f = findMax(datos)
    text = "Prediction " + str(x) + ": "
    for i in f:
        text = text +" - "+ categories[i[0]]
    print(text)

def plot_scenes(pred,title):
    plt.figure(figsize=(16,8))
    for i in range(pred.shape[1]):# number of prediction vectors
        datos = (pred[0,i,:] > 0)* pred[0,i,:] #find classes
        label='Window no: '+ str(i)
        plt.bar(np.arange(0,pred.shape[2]),datos,label = label) # it draws the histogram for each window
        m = np.argmax(datos)
        plt.text(m,max(datos)-(0.01),'Scene out: '+ str(i) +categories[m])
        putText(datos,i)

    plt.legend()
    plt.title(title)
    plt.xlabel('Scenes')
    plt.show()

# Use previous function to show histogram in this first audio
#plot_scenes(prediction,'Output railroad')
#
#
#
# def plot_objects(pred,title):
#     plt.figure(figsize=(16,8))
#     for i in range(pred.shape[1]):# number of prediction vectors
#         datos = (pred[0,i,:] > 0)* pred[0,i,:] #find classes
#         label='Window no: '+ str(i)
#         plt.bar(np.arange(0,pred.shape[2]),datos,label = label) # it draws the histogram for each window
#         m = np.argmax(datos)
#         plt.text(m,max(datos)-(0.01),'Scene out: '+ str(i) +image_objectts[m])
#         putText(datos,i)
#
#     plt.legend()
#     plt.title(title)
#     plt.xlabel('Objects')
#     plt.show()
#
#
# plot_objects(prediction,'Output railroad')




#print(prediction)

# Repeat the process to obtain a new prediction
#prediction2 = predict_scene_from_audio_file('/Users/christos/PycharmProjects/pythonProject/SoundNet-keras/soundnet_keras-master/school.wav')
#print("output model shape", prediction2.shape) #output model
#
# plot_objects(prediction2,'Output school yard')



print ('LETS MOVE TO THE CUSTOM PART')



import pandas as pd
test = pd.read_csv('/Users/christos/PycharmProjects/pythonProject/SoundNet-keras/ESC-50-master/meta/esc50.csv',sep=',')
#test.head()
#
# esc10 = test[test['esc10'] == False]   #### dokimazw gia 40 classes
# print(esc10[:5]) #first 5 elements
esc10 = test

data = [] #load the audiofiles
data2 = []
list_target = esc10['target'] #list of targets in dataset ESC10
list_category = esc10['category'] #list of categories (scenes) in dataset ESC10




#loop to load audiofiles
for file_name in list(esc10['filename']):
    audio = load_audio('/Users/christos/PycharmProjects/pythonProject/SoundNet-keras/ESC-50-master/audio/'+ file_name)
    data.append(audio)
    print(len(data))



for j in range(0,len(data)):
    print('the shape of this element is',data[j].shape)


print(type(data))
datos = np.asarray([data[155]]).reshape(1,-1,1) # failure due to the size
print("Data shape: ",datos.shape)
#
# datos = np.asarray([data[5],data[5],data[5]]).reshape(1,-1,1)
# print("Input shape: ", datos.shape)
#
# p = model.predict(datos)
# print('p is', p.shape)



#- > Three times data: enough samples for the model
#print("Output shape: ",p.shape)
#plot_scenes(p,'output esc10-helicopter')

from keras import backend as K


# return the list of activations as tensors for an specific layer in the model
def getActivations(data, number_layer):
    intermediate_tensor = []
    # get Hidden Representation function
    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[number_layer].output])
    # just for information
    ex = get_layer_output([data[155]])[0]
    print('Dimension layer {}: {}'.format(number_layer, ex.shape))

    for audio in data:
        # get Hidden Representation
        layer_output = get_layer_output([audio])[0]  # multidimensional vector
        print(layer_output.shape,'layer output')
        tensor = layer_output.reshape(1, -1)  # change vector shape to 1 (tensor)
        print(tensor.shape,'tensor')
        intermediate_tensor.append(tensor[0])  # list of tensor activations for each object in Esc10

    return intermediate_tensor



activations22 = getActivations(data,22) #get activation tensor for the 22nd layer in the model (pool5)
print(activations22)

layers = [0,4,9,15,22,28] # List of important hidden layers

#obtain vector layer 22 INPUT
x = np.asarray(getActivations(data,22)) #Transfer learning
print('Dataset input shape', x.shape)
y = np.asarray(list_target)
#print('Target list shape',y.shape)



#obtain original values
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



from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
num_classes = 50



#Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state = 25)
z_train = to_categorical(y_train, num_classes) # convert class vectors to binary class matrices
z_test = to_categorical(y_test, num_classes)


from keras.models import Sequential
from keras.layers import Dense, Flatten



classifier = Sequential()
classifier.add(Dense(num_classes, activation='softmax',input_shape=(3328,)))
classifier.summary()

from keras.optimizers import Adam
classifier.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])

batch_size =64
epochs = 100
#fit classifier
hist_classifier = classifier.fit(x_train,z_train, validation_data=(x_test,z_test), epochs=epochs,batch_size=batch_size)



classifier.save("testmodel_50.h5")



def evaluate_model(model,history,title):
    print(history.history.keys())
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy '+ title)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train','val'], loc='upper left')
   # plt.show()

    score = model.evaluate(x_test, z_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

evaluate_model(classifier,hist_classifier,'softmax')

x_pred = classifier.predict(x_test)
x_pred_classes = x_pred.argmax(axis=-1)





# print(x_test)
#
# for i in range(len(x_test)):
#     #print(x_pred[i], toCategory[x_classes[i]])
#     print(x_test[i], toCategory[x_pred_classes[i]])
#
# ######## for a custom folder
