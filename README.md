![1-s2 0-S2772528622000280-gr004](https://user-images.githubusercontent.com/106226707/212156906-8196a6f6-0bee-4c20-8248-5e1103533d72.jpg)

# Classification of Alzheimer's Disease using MRI Images
Project status: `Completed`

Authors:

[Goretti Muthoni](https://github.com/Gorreti)

[Jeremy Nguyo](https://github.com/NguyoJer)

[Getrude Obwoge](https://github.com/Getty3102)

[Jane NJuguna](https://github.com/janejeshen)

[Abduba Galgalo]()

[Amos Pride](https://github.com/amoskiito)

[Collins Kiptoo](https://github.com/Collins-Kiptoo)

# Project Goal

This project is designed to classify the stage of Alzheimer's disease in patients by analyzing brain scan images. The model was trained on a dataset of brain scans and uses a convolutional neural network (CNN) to make predictions.
Alzheimer's Disease is a progressive disorder that destroys memory and other important mental functions. It is the most common cause of dementia among older adults. The main aim of creating this app is to classify the stage of dementia for patients already diagnosed with the Alzheimer's disease.

# Data Understanding
We used two dataset in our project, the first one was used for explolatory data analysis and the second was used for modelling

The first dataset has 436 rows and 11 columns.

The second dataset consists of MRI images of the brain of individuals diagnosed with Alzheimer's disease. The dataset is divided into a training set and a test set. The dataset contains 6400 MRI images.

## Preprocessing:
The MRI images were preprocessed to standardize their size and intensity levels. The images were resized to 150x150 pixels.

## Data source: 
The datasets were sourced from kaggle [first dataset](https://www.kaggle.com/code/obrienmitch94/alzheimer-s-analysis/data) and [second dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)

## Technologies Used:
* python 
* pandas
* matplotlib
* scikit-learn
* Tensorflow
* Streamlit
* Seaborn

## Models Used:
* VGG16
* VGG19
* EfficientNetB0
* DenseNet
* Sequential

## Best model:

The best model was a convolutional neural network.The model was trained using the Adam optimizer and a categorical cross-entropy loss function. The model was evaluated using metrics such as accuracy, precision and  recall.
The model achieved an accuracy of 95%, a precision of 95%, and a recall of 95%. These results indicate that the model is able to accurately classify the level of dimentia on individuals diagnosed with Alzheimer's disease

## Limitations:

This project has some limitations. The dataset used in this project is small and further research is needed to validate the results on larger datasets. 

## How to use the codes:
The code for this project can be run using Python 3 and TensorFlow tensorflow==2.3.1. To use the model, you will need to install the necessary dependencies and input new MRI images for classification. The instructions for using the model are provided in the code.
