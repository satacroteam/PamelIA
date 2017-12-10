'''

Allows to predict the probability to belongs at each classes on which the model has trained

'''
import os, time
import numpy as np
import pickle
import re

import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score
from keras.models import load_model
from keras.preprocessing import image


def predict(img_path, prediction_model):
    """
    Predict probability for one picture
    :param img_path: Path of the image
    :param prediction_model: Model to use for prediction
    :return: The prediction (tuple)
    """
    # Load image and resize it
    img = image.load_img(img_path, target_size=(224, 224))
    # Transform it in array
    x = image.img_to_array(img)
    # Expand array dimension
    x = np.expand_dims(x, axis=0)
    # Make prediction
    prediction_score = prediction_model.predict(x)
    return prediction_score


if __name__ == '__main__':
    # Create the correct result list
    result = []
    result = pd.read_csv('result_test.csv', sep=',')['2'].tolist()
    # Define the prediction list
    result_pred = []

    # Define the test directory PATH
    test_directory_path = 'test'

    # If the preictions have not been done
    i = 0
    if not os.path.isfile('result_pred'):

        # Load the model
        model_path = r'InceptionV3_final.h5'
        print('Loading model:', model_path)
        t0 = time.time()
        model = load_model(model_path)
        t1 = time.time()
        print('Loaded in:', t1 - t0)

        print("Calculate prediction : ")

        # Predict the answer probabilities for all the pictures of the test dataset
        for image_name in os.listdir(test_directory_path):
            i += 1
            print(i)
            test_path = os.path.join(test_directory_path, image_name)
            preds = predict(test_path, model)
            image_name = re.sub(r'\.jpg$', '', image_name)
            result_pred.append(preds[0][0])

        # Save the model
        with open("result_pred", "wb") as result_pred_file:
            pickle.dump(result_pred, result_pred_file)

        # Print confusion matrix and mean average precision score
        result_pred_binary = [int(round(answer)) for answer in result_pred]
        print("\nConfusion matrix : ")
        print(confusion_matrix(result, result_pred_binary))
        print("\nAverage precision score : ", average_precision_score(result, result_pred_binary))
    else:
        # Print confusion matrix and mean average precision score
        print("\n\nLoad prediction answers : ")
        with open("result_pred", "rb") as result_pred_file:
            result_pred = pickle.load(result_pred_file)
        result_pred_binary = [int(round(answer)) for answer in result_pred]
        print("\nConfusion matrix : ")
        print(confusion_matrix(result, result_pred_binary))
        print("\nAverage precision score : ", average_precision_score(result, result_pred_binary))
