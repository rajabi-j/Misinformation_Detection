# Misinformation Detection


## Description

### Co_Occurence

Co_Occurence.ipynb file extract the events and vocabularies from the tweets and calculates the co_occurance matrices.


### IVA

IVA_G_SPICE file runs IVA_G and IVA_SPICE on different range of data dimensions (from 7 to 19).


### Covariance Matrices and Training SVM Models

Covariance_withTraining.py calculate covariance matrices based on IVA S matrices. 
Then, it finds the best SVM model using cross-validation.


### A Matrices and Training SVM Models

Save_Event_Vocabs.ipynb file Stores events vocabularies to be used in A matrices.

Amatrices_withTraining.py calculate A matrices based on IVA output. 
Then, it finds the best SVM model using cross-validation.


### Average Matrices and Training SVM Models

AverageMatrices_withTraining.py calculate average matrices based on IVA S matrices. 
Then, it finds the best SVM model using cross-validation.


## Results

Results_Covariance_Matrices.ipynb shows the test results for Covariance Matrices.

Results_A_Matrices.ipynb shows the test results for A Matrices.

Results_Average_Matrices.ipynb shows the test results for Average Matrices.

Events.ipynb shows the events with highest vocabulary Size.

IVA_Time.ipynb shows the time spent on running different IVAs
