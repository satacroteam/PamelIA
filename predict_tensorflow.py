"""
Allows to predict the probability to belongs at each classes on which the model has trained
"""
import os, time
import numpy as np
import pickle

import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score
from keras.models import load_model
from keras.preprocessing import image


class Predict(object):
    """
    Class to predict
    """
    def __init__(self):
        # Define the test directory PATH
        self.test_directory_path = 'test'

        # Define the correct answer csv
        self.result = pd.read_csv('result_test.csv', sep=',')['2'].tolist()

        # Define the predicted results and predicted results binary list
        self.predicted_results = []
        self.predicted_results_binary = []

        # Define the model to use
        self.model_path = 'InceptionV3_final.h5'

    def load_model(self):
        """
        Load the model define in init
        :return: The model object
        """
        # Load the model
        print('Loading model:', self.model_path)
        t0 = time.time()
        model = load_model(self.model_path)
        t1 = time.time()
        print('Loaded in:', t1 - t0)
        return model

    @staticmethod
    def predict_one_image(img_path, prediction_model):
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

    @staticmethod
    def predicted_results_to_binary(predicted_results):
        """
        Build binary predicted results list with the predicted results
        :param predicted_results: Predicted results list
        :type predicted_results: List
        :return: Binary predicted results list
        """
        return [int(round(answer)) for answer in predicted_results]

    def statistics_on_test(self, predicted_results, result):
        """
        Print confusion matrix and average precision score
        :param predicted_results: Predicted results list
        :param result: True results list
        """
        # Print confusion matrix and mean average precision score
        predicted_results_binary = self.predicted_results_to_binary(predicted_results)
        print("\nConfusion matrix : ")
        print(confusion_matrix(result, predicted_results_binary))
        print("\nAverage precision score : ", average_precision_score(result, predicted_results_binary))

    @staticmethod
    def save_predicted_results(predicted_results):
        """
        Save predicted results as pickle
        :param predicted_results: Predicted results list
        """
        # Save the model
        with open("predicted_results", "wb") as predicted_results_file:
            pickle.dump(predicted_results, predicted_results_file)

    def load_predicted_results(self):
        """
        Load predicted results from pickle
        """
        print("\n\nLoad prediction answers : ")
        with open("predicted_results", "rb") as predicted_results:
            self.predicted_results = pickle.load(predicted_results)

    def predict(self):
        """
        If not predicted_results:
            - Predict probability for all test directory images
            - Fill predicted_results with them
            - Save predicted_results as pickle

        If predicted_results:
            - Load predicted_results from pickle

        Show statistics (confusion matrix and average precision score)
        """
        # If the preictions have not been done
        i = 0
        if not os.path.isfile('predicted_results'):

            # Load the model
            model = self.load_model()

            print("Calculate prediction : ")

            # Predict the answer probabilities for all the pictures of the test dataset
            for image_name in os.listdir(self.test_directory_path):

                # Show where we are
                i += 1
                print(i)

                # Define the path for the image
                test_path = os.path.join(self.test_directory_path, image_name)

                # Make the prediction
                image_prediction = self.predict_one_image(test_path, model)

                # Append the prediction to the predicted results list
                self.predicted_results.append(image_prediction[0][0])

            # Save predicted results
            self.save_predicted_results(self.predicted_results)

        else:
            # Load the predicted results list
            self.load_predicted_results()

        # Print confusion matrix and mean average precision score
        self.statistics_on_test(self.predicted_results, self.result)

    # if __name__ == '__main__':
    #    predict()
