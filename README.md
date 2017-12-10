# PamelIA
Data Science project to explore the integration of IA in the medical field

## Création du model
Launch this file : train_tensorflow.py
It will create two files (best and final) with the model.

## App
predict_tensorflow.py {NOM_MODEL.h5} {NOM_PHOTO}

## Project tree
IASC
 * [data](IASC/data)
   Data to train the model
   * [train](IASC/data/train)
     *[benign](IASC/data/train/benign)
     Pictures of your first category
     *[malignant](IASC/data/train/malignant)
     Pictures of your second category
   Data to validate the training
   * [valid](IASC/data/valid)
     *[benign](IASC/data/train/benign)
     Pictures of your first category
     *[malignant](IASC/data/train/malignant)
     Pictures of your second category
 Data to test the model
 * [test](IASC/test)


