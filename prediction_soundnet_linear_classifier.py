from glob import glob
import keras
import os
from keras.models import load_model
import numpy as np
import librosa
import pandas as pd
import argparse

test = pd.read_csv('/Users/christos/PycharmProjects/pythonProject/SoundNet-keras/ESC-50-master/meta/esc50.csv',sep=',')

categories_full = test['category']
categories = []

for i in range(len(categories_full[:])):
    if categories_full[i] not in categories:
        categories.append(categories_full[i])
fn = ""



def load_pretrained():
    #fn = "aek"
    model = load_model(fn)
    #model.summary()
    return model


def preprocess(audio):
    audio *= 256.0  # SoundNet needs the range to be between -256 and 256
    # print(audio.shape,'prin')
    audio = np.reshape(audio, (1,-1,1)) #### edw einai to lathos
    audio = np.resize(audio,(1,3328))
    # print(audio.shape,'meta')
    return audio


def load_audio(audio_file):
    sample_rate = 22050  # SoundNet works on mono audio files with a sample rate of 22050.
    audio, sr = librosa.load(audio_file, dtype='float32', sr=22050, mono=True)
    audio = preprocess(audio)
    return audio


def define_all_predictions(prediction):
    ten_predicted_scenes = []
    sorted_categories = np.argsort(prediction[0])[:-11:-1]
    for i in range(len(sorted_categories)):
        ten_predicted_scenes.append((categories[sorted_categories[i]]))

    return ten_predicted_scenes

def predict_scene_from_audio_file(audio_file):
    model = load_pretrained()
    audio = load_audio(audio_file)
    prediction = model.predict(audio)
    return prediction


parser = argparse.ArgumentParser()
parser.add_argument('Load_folder', metavar='Load_folder', type=str,
                    help='specify the folder you want to test, full path')
parser.add_argument('Load_file', metavar='Load_file', type=str,
                    help='specify the saved model to load')


args= parser.parse_args()

base_path = args.Load_folder
fn = args.Load_file
print("the model you chose is", fn)


if __name__ == '__main__':
    output_path = '/Users/christos/PycharmProjects/pythonProject/Soundnet_predictions_finetuned'+os.sep
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(base_path, 'base path is')
    test_files = glob(base_path+'*wav')+glob(base_path+'*mp3')
    for f in test_files:
        #print(f)
        prediction = predict_scene_from_audio_file(f)
        #print(prediction.shape)
        # predicted_objects = predictions_to_objects(prediction)
        # predicted_scenes = predictions_to_scenes(prediction)
        # print ('Scene is',predicted_scenes)

        predicted_ten_scenes = define_all_predictions(prediction)
        print('Ten probable scenes are', predicted_ten_scenes)
        # print('Object is', predicted_objects)
        print('----')
        output_filename = f.split(os.sep)[-1].split('.')[0]

      

        with open(output_path+output_filename+'.csv','w') as prediction_file:
            prediction_file.write('filename:'+ str(f)+ '\n')
            prediction_file.write('predicted_vector:'+ str(prediction) + '\n')
            # prediction_file.write('predicted_objects:' + str(predicted_objects)+'\n')
            prediction_file.write('predicted_scenes:'+ str(predicted_ten_scenes))

        prediction_file.close()
