from glob import glob
import keras
import os
from keras.models import load_model
import numpy as np
import librosa
import pandas as pd


test = pd.read_csv('/Users/christos/PycharmProjects/pythonProject/SoundNet-keras/ESC-50-master/meta/esc50.csv',sep=',')

categories_full = test['category']
categories = []

for i in range(len(categories_full[:])):
    if categories_full[i] not in categories:
        categories.append(categories_full[i])

# #[11 26 46 29 12  4 49 21 27 28]
# for j in range(len(pred_list)):
#     results =



def load_pretrained():
    fn = "testmodel_50.h5"
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



# def predictions_to_scenes(prediction):
#     scenes = []
#     for p in range(len(prediction)):
#             scenes.append(categories[np.argmax(prediction[p, :])])
#             print(scenes)
#
#     return scenes



def define_all_predictions(prediction):
    final_list = []
    ten_predicted_scenes = []
    sorted_categories = np.argsort(prediction[0])[:-11:-1]
    #print(sorted_categories)
    for i in range(len(sorted_categories)):
        ten_predicted_scenes.append((categories[sorted_categories[i]]))

    return ten_predicted_scenes




def predict_scene_from_audio_file(audio_file):
    model = load_pretrained()
    audio = load_audio(audio_file)
    prediction = model.predict(audio)
    return prediction



if __name__ == '__main__':
    output_path = '/Users/christos/PycharmProjects/pythonProject/Soundnet_predictions_finetuned'+os.sep
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    # base_path = '../ESC-50-master/audio/'
    base_path = '/Users/christos/PycharmProjects/pythonProject/scenesNoisy03-Jun-2022/'
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

        # d={}
        # d['filename'] = f
        # d['predicted_vector'] = prediction
        # d['predicted_objects'] = predicted_objects
        # d['predicted_scenes'] = predicted_scenes
        # # prediction_file.write('predicted_scenes:'+ str(predicted_scenes))
        # pd.DataFrame.from_dict(d)
        # d.to_csv(output_path + output_filename)

        with open(output_path+output_filename+'.csv','w') as prediction_file:
            prediction_file.write('filename:'+ str(f)+ '\n')
            prediction_file.write('predicted_vector:'+ str(prediction) + '\n')
            # prediction_file.write('predicted_objects:' + str(predicted_objects)+'\n')
            prediction_file.write('predicted_scenes:'+ str(predicted_ten_scenes))

        prediction_file.close()

