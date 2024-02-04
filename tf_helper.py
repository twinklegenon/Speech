import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras import models  


loaded_model = models.load_model("speech.hdf5")

def features_extractor(audio):
    # load the file (audio)
    audio, sample_rate = librosa.load(file_name, res_type='fft')
    # extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    # in order to find out scaled feature we do the mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


def preprocess_and_predict(audio):
    # Extract features from the audio
    features = features_extractor(audio)
    # Expand dimensions to match the input shape expected by the CNN model
    features = np.expand_dims(features, axis=0)

    # Make prediction using the CNN model
    predictions = loaded_model.predict(features)

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)

    # Map the predicted index to the class label
    class_labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']  # Replace with your actual class labels
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label
