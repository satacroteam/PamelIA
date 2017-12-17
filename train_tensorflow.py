"""
Allows to re-train the model with a new multi-classes dataset
You could choose between 3 pre-trained models : ResNet50, InceptionResNetV2 et InceptionV3
"""
import os
import math

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image


class Train(object):
    """
    Class to train the model
    """
    def __init__(self):
        self.DATA_DIR = 'data'
        self.TRAIN_DIR = os.path.join(self.DATA_DIR, 'train')
        self.VALID_DIR = os.path.join(self.DATA_DIR, 'valid')
        self.SIZE = (224, 224)
        self.BATCH_SIZE = 16
        self.AVAILABLE_MODEL = ['InceptionV3', 'ResNet50']
        self.MODEL = 'InceptionV3'
        self.MODEL_NAME_FINAL = self.MODEL+'_final.h5'
        self.MODEL_NAME_BEST = self.MODEL+'_best.h5'
        self.EPOCHS = 10
        self.LOSS = 'categorical_crossentropy'
        self.METRICS = ['accuracy']
        self.ACTIVATION = "softmax"

    @staticmethod
    def num_samples(directory_path):
        """
        Find number of files in a directory
        :param directory_path: Path to the directory
        :return: The number of files
        """
        return sum([len(files) for r, d, files in os.walk(directory_path)])

    def num_steps(self, directory_path, batch_size):
        """
        Find number of steps for the model
        :param directory_path: Path to the directory
        :param batch_size: Batch size
        :return: The number of steps
        """
        num_train_samples = self.num_samples(directory_path)
        return math.floor(num_train_samples / batch_size)

    @staticmethod
    def image_flow_generator(directory_path, image_size, batch_size):
        """

        :param directory_path:
        :param image_size:
        :param batch_size:
        :return:
        """
        # Define the generator of image flow for the train and valid repository
        generator = keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True
        )

        # Define the generator of image flow for the train and valid repository
        image_flow_generator = generator.flow_from_directory(
            # Train directory path
            directory_path,
            # Size of the picture
            target_size=image_size,
            # Class mode
            class_mode='categorical',
            # Shuffling of data
            shuffle=True,
            # Batch size
            batch_size=batch_size
        )
        return image_flow_generator

    @staticmethod
    def choose_model(model_name):
        """
        Choice of the pre-trained model
        :return: The pre-trained model
        """
        model = None
        if model_name == 'ResNet50':
            model = keras.applications.resnet50.ResNet50()
        elif model_name == 'InceptionV3':
            model = keras.applications.inception_v3.InceptionV3()
        return model

    def train(self):
        # Define the steps for the train and valid directory
        num_train_steps = self.num_steps(self.TRAIN_DIR, self.BATCH_SIZE)
        num_valid_steps = self.num_steps(self.VALID_DIR, self.BATCH_SIZE)

        # Define the generator of image flow for the train and valid repository
        train_image_flow_generator = self.image_flow_generator(self.TRAIN_DIR, self.SIZE, self.BATCH_SIZE)
        valid_image_flow_generator = self.image_flow_generator(self.VALID_DIR, self.SIZE, self.BATCH_SIZE)

        # Choice of the pre-trained model
        model = self.choose_model(self.MODEL)

        # Define the list of classes (number of repository in the train folder)
        classes = list(iter(train_image_flow_generator.class_indices))

        # Delete the last layer of the neural network
        model.layers.pop()

        # Fix all the layers to a none trainable state
        for layer in model.layers:
            layer.trainable = False

        # Get the output attribute of the last layer (previous before last) of the neural net
        last = model.layers[-1].output

        # Link it to a new layer with the number of output corresponding to the train dataset
        x = Dense(len(classes), activation=self.ACTIVATION)(last)

        # Link this new layer to the input (as a new Model)
        fine_tuned_model = Model(model.input, x)

        # Compile it
        fine_tuned_model.compile(
            optimizer=Adam(lr=0.0001),
            loss=self.LOSS,
            metrics=self.METRICS
        )

        # Define classes
        for c in train_image_flow_generator.class_indices:
            classes[train_image_flow_generator.class_indices[c]] = c

        # Add it to the model
        fine_tuned_model.classes = classes

        # Define early stopping variable
        early_stopping = EarlyStopping(patience=10)

        # Define check pointer variable
        check_pointer = ModelCheckpoint(
            self.MODEL_NAME_BEST,
            verbose=1,
            save_best_only=True
        )

        # Fit the model
        fine_tuned_model.fit_generator(
            train_image_flow_generator,
            steps_per_epoch=num_train_steps,
            epochs=self.EPOCHS,
            callbacks=[early_stopping, check_pointer],
            validation_data=valid_image_flow_generator,
            validation_steps=num_valid_steps
        )

        # Save final model
        fine_tuned_model.save(self.MODEL_NAME_FINAL)

    if __name__ == "__main__":
        train()
