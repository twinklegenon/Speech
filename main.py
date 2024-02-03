import numpy as np
import tensorflow as tf
from tensorflow.keras import models  # This is the correct import for models

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer

# !! Modify this in the correct order
x_labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

loaded_model = models.load_model("speech.hdf5")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)

    # Reshape spec to match the expected input shape of the model
    spec = tf.image.resize(spec, (60, 1))  # Adjust dimensions as needed

    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = x_labels[label_pred[0]]
    print("Predicted label:", command)
    return command

if __name__ == "__main__":
    from turtle_helper import move_turtle
    while True:
        command = predict_mic()
        move_turtle(command)
        if command == "stop":
            terminate()
            break