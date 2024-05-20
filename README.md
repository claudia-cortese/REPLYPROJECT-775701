# **RESUME CLASSIFICATION**

## INTRODUCTION

This project aims to build a model that accurately identifies the class to which each resume belongs.

The dataset is made of 5375 black and white images of scanned resumes in .tif format that will be classified into four categories.

The four classes identified are:

- academic resumes, characterised by a dominance of the education section which is detailed and placed on top of the cv;
- industrial resumes, that present a prominence on work experience, extensive details of professional roles and at the same time with less attention given to academic achievements;
- research resumes, that include meticulous descriptions of research projects and methodologies and comprehensive lists of publications and patents;
- leadership resumes, that emphasize sections that include leadership roles, business metrics and keynote speeches.

The goal is to utilize one or more models to identify the various types of images within the dataset.

The steps we followed are: 

1) exploratory data analysis, to first inspect the dataset. Since our dataset is made of images, the features analyzed were widht, height, file size, contrast, brightness and noise level. This phase is pivotal for all the next steps of the project.
2) preprocessing, to prepare the data for modelling. The actions taken were resizing, pixel normalization, conversion from images to numpy array, extraction of LBP features and splitting into train and test sets. It is followed by a principal component analysis to reduce the dimensionality of the features, remove noise and redundancy.
3) modelling, to classify the resumes into four classes. The chosen models are: k-means, ANN, autoencoder and variational autoencoder (VAE). We performed extensive validation through cross-validation and analysis of learning curves to ensure the robustness and generalizability of the models.



## METHODS




