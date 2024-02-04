import numpy as np
import tensorflow as tf
from recording_helper import record_audio, terminate
from tf_helper import preprocess_and_predict


def predict_mic():
    audio = record_audio()
    command = preprocess_and_predict(audio)
    print(f'The predicted class label for the audio file is: {command}')
    return command

if __name__ == "__main__":
    from turtle_helper import move_turtle
    while True:
        command = predict_mic()
        move_turtle(command)
        if command == "stop":
            terminate()
            break
