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
3) modelling, to classify the resumes into four classes. The chosen models are: k-means, ANN, autoencoder and Variational Autoencoder (VAE). We performed extensive validation through cross-validation and analysis of learning curves to ensure the robustness and generalizability of the models.



## METHODS

Our work focused on the combination of both unsupervised and supervised learning techniques to effectively classify and evaluate image data. Before delving into feature extraction, algorithms adopted and training overview, let's set up the environment.
	
	
#### Environment Setup
This project was entirely created on Google Colab. If you'd like to use Google Colab, just run the following line at the beginning of the notebook to install the required libraries.


```bash
!pip install matplotlib pandas numpy seaborn tensorflow scikit-image pillow joblib scikit-learn keras
```

Then, in order to ensure compatibility, you can run the command 
```bash
!pip freeze
```
to check the versions of the libraries.

Here are the libraries and their respective versions:

	matplotlib==3.3.4
	pandas==1.1.5
	numpy==1.19.5
	seaborn==0.11.1
	tensorflow==2.4.1
	joblib==1.0.1
	Pillow==8.1.0
	scikit-image==0.18.1
	scikit-learn==0.24.2
	keras==2.4.3


If you're not using Google Colab, you might want to clone the repository to your local machine:

```bash
git clone https://github.com/claudia-cortese/REPLYPROJECT-775701.git
```
Then, navigate to the repository directory:
```bash
cd REPLYPROJECT-775701
```
 Install the required libraries using pip:
```bash
pip install -r requirements.txt
```
Alternatively, if you are using Conda, you can create an environment using the `environment.yml` file:
  ```bash
  conda env create -f environment.yml
  conda activate resume_classification
  ```
Finally, run the Jupyter Notebook:
```bash
jupyter notebook
```

#### Flowchart

Now that the environment s set up, we can procede to describe our ideas and the design choices. In order to do so, we begin with a flowchart that will help us visualize the steps we followed.

	+----------------------------+
			| Load & Preprocess          |
			| Data (Images)              |
	+----------------------------+
            |
            v
	+----------------------------+
			| Feature Extraction         |
			| (LBP, PCA, VAE)            |
	+----------------------------+
            |
            v
	+----------------------------+
			| Unsupervised Learning      |
			| (K-Means Clustering)       |
	+----------------------------+
            |
            v
	+----------------------------+
			| Supervised Learning        |
			| (ANN Training)             |
	+----------------------------+
            |
            v
	+----------------------------+
			| Model Evaluation           |
			| (Metrics & Validation)     |
	+----------------------------+



#### Feature Extraction
Now that the environnment is set up, we can 
- **Local Binary Patterns (LBP)**: A simple yet efficient texture operator that labels the pixels of an image by thresholding the neighborhood of each pixel and considering the result as a binary number.
- **Principal Component Analysis (PCA)**: A dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming the data to a new coordinate system such that the greatest variance by some projection of the data comes to lie on the first coordinates (called the principal components).
- **Variational Autoencoders (VAE)**: A generative model that leverages neural networks to learn the underlying distribution of data in a lower-dimensional space and can be used to generate new data points.

#### Algorithms
- **K-Means Clustering**: An unsupervised learning algorithm used to partition the dataset into K distinct, non-overlapping subsets (clusters).
- **Artificial Neural Networks (ANN)**: A type of deep learning model that is particularly effective for tasks involving image data, used here for supervised learning to classify résumés.

#### Training Overview
- **Data Normalization**: Adjusting the values in the dataset to a common scale, typically ranging from 0 to 1, without distorting differences in the ranges of values.
- **Data Augmentation**: Techniques such as rotations, zooming, and horizontal flips applied to training data to increase the diversity of data available for training models, thereby improving the generalization ability of the models.
- **Hyperparameter Tuning**: The process of optimizing the parameters that govern the training process of machine learning models, such as learning rate, number of layers, and batch size.



### Flowchart




