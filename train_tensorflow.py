'''

Allows to re-train the model with a new multi-classes dataset

You could choose between 3 pre-trained models : ResNet50, InceptionResNetV2 et InceptionV3

'''
import math, os

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image

################################################################################################
#################################### CONFIG ####################################################
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
SIZE = (224, 224)
BATCH_SIZE = 16
AVAILABLE_MODEL = ['InceptionV3', 'InceptionResNetV2', 'ResNet50']
MODEL = 'InceptionV3'
MODEL_NAME_FINAL = MODEL+'_final.h5'
MODEL_NAME_BEST = MODEL+'_best.h5'
EPOCHS = 10
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
ACTIVATION = "softmax"
################################################################################################
################################################################################################

if __name__ == "__main__":

    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    # Define the generator of image flow for the train and valid repository
    gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    # Define the generator of image flow for the train and valid repository
    batches = gen.flow_from_directory(
                            # Train directory path
                            TRAIN_DIR,
                            # Size of the picture
                            target_size=SIZE,
                            # Class mode
                            class_mode='categorical',
                            # Shuffling of data
                            shuffle=True,
                            # Batch size
                            batch_size=BATCH_SIZE
    )

    val_batches = val_gen.flow_from_directory(
                            # Valid directory path
                            VALID_DIR,
                            # Size of the picture
                            target_size=SIZE,
                            # Class mode
                            class_mode='categorical',
                            # Shuffling of data
                            shuffle=True,
                            # Batch size
                            batch_size=BATCH_SIZE
    )

    # Choice of the pre-trained model
    model = None
    if MODEL == 'ResNet50':
        model = keras.applications.resnet50.ResNet50()
    elif MODEL == 'InceptionResNetV2':
        model = keras.applications.inception_resnet_v2.InceptionResNetV2()
    elif MODEL == 'InceptionV3':
        model = keras.applications.inception_v3.InceptionV3()

    # Define the list of classes (number of repository in the train folder)
    classes = list(iter(batches.class_indices))

    # Delete the last layer of the neural network
    model.layers.pop()

    # Fix all the layers to a none trainable state
    for layer in model.layers:
        layer.trainable = False

    # Get the output attribute of the last layer (previous before last) of the neural net
    last = model.layers[-1].output

    # Link it to a new layer with the number of output corresponding to the train dataset
    x = Dense(len(classes), activation=ACTIVATION)(last)

    # Link this new layer to the input (as a new Model)
    finetuned_model = Model(model.input, x)

    # Compile it
    finetuned_model.compile(
        optimizer=Adam(lr=0.0001),
        loss=LOSS,
        metrics=METRICS
    )

    # Define classes
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c

    # Add it to the model
    finetuned_model.classes = classes

    # Define early stopping variable
    early_stopping = EarlyStopping(patience=10)

    # Define check pointer variable
    check_pointer = ModelCheckpoint(
        MODEL_NAME_BEST,
        verbose=1,
        save_best_only=True
    )

    # Fit the model
    finetuned_model.fit_generator(
        batches,
        steps_per_epoch=num_train_steps,
        epochs=EPOCHS,
        callbacks=[early_stopping, check_pointer],
        validation_data=val_batches,
        validation_steps=num_valid_steps
    )

    # Save final model
    finetuned_model.save(MODEL_NAME_FINAL)