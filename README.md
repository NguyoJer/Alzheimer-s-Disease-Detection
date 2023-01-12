![img](.https://images.app.goo.gl/P5AszgC3qoT1cjRN7)

# IMAGE CLASSIFICATION
# Alzheimers Disease Classification Using MRI images
Project status: `Completed`

Authors:

[Goretti Muthoni](https://github.com/Gorreti)

[Jeremy Nguyo](https://github.com/NguyoJer)

[Getrude Obwage](https://github.com/Getty3102)

[Jane NJuguna](https://github.com/janejeshen)

[Abduba Galgalo]()

[Amos Pride](https://github.com/amoskiito)
[Collins KIptoo](https://github.com/Collins-Kiptoo)

# Business Understanding
## Project Goal
The goal of the project is to develop a method for classifying Alzheimer's disease using MRI images. This will involve analyzing the structural changes in the brain that occur as a result of the disease and using machine learning techniques to accurately identify and classify individuals with Alzheimer's. The ultimate goal is to improve the accuracy and speed of diagnosing Alzheimer's disease, which will aid in early detection and treatment.
## Overview
## Problem Statement

# Data Understanding
The first dataset has 436 rows and 11 columns.

The second data set contains 6400 entries consisting of mri images belonging to 4 classes
## Column Descriptions:

Subject.ID: Unique identifier for the Alzheimer's patients.

MRI.ID: Unique identifier for the MRI scans

Group (Converted / Demented / Nondemented): The column tells whether the Alzheimers patient is demented or not also whether the patient was the patient was non demented but was diagnosed as demented after a couple of visits.

Visit- Number of visit: The number of times the patient

MR.Delay

`Demographics Info`

M.F - Gender of the patient.

Hand - Handedness (actually all subjects were right-handed so I will drop this column)

Age - Age of the patient

EDUC - Years of education

SES - Socioeconomic status as assessed by the Hollingshead Index of Social Position and classified into categories from 1 (highest status) to 5 (lowest status)

`Clinical Info`

MMSE - Mini-Mental State Examination score (range is from 0 = worst to 30 = best)

CDR - Clinical Dementia Rating (0 = no dementia, 0.5 = very mild dementia, 1 = mild dementia, 2 = moderate dementia)

Derived Anatomic Volumes

eTIV - Estimated total intracranial volume, mm3

nWBV - Normalized whole-brain volume, expressed as a percent of all voxels in the atlas-masked image that are labeled as gray or white matter by the automated tissue segmentation process

ASF - Atlas scaling factor (unitless). Computed scaling factor that transforms native-space brain and skull to the atlas target (i.e., the determinant of the transform matrix)

## Data source: 

The data was sourced from kaggle [website](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)



## Alzheimers Disease Classification Using MRI images

** Project status: ** Completed

## Dataset
The data set contains 6400 entries consisting of mri images belonging to 4 classes

## Data source: 
The data was sourced from kaggle [website](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)

## Technologies:
* python 
* pandas
* matplotlib
* scikit-learn
* statsmodel
* Tensorflow
* Streamlit
* Seaborn
