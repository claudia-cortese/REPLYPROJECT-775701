# **RESUME CLASSIFICATION**

####Team Members: Rachele Cecere (775701), Claudia Cortese (785561)

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
2) preprocessing, to prepare the data for modelling. The actions taken were resizing, pixel normalization, conversion from images to numpy array, extraction of LBP features and splitting into train and test sets. It was followed by a principal component analysis (PCA) to reduce the dimensionality of the features, remove noise and redundancy.
3) modelling, to classify the resumes into four classes. The chosen models are: **k-means**, **ANN**, **autoencoder** and **Variational Autoencoder (VAE)**. We performed extensive validation through cross-validation and analysis of learning curves to ensure the robustness and generalizability of the models.



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

Now that the environment is set up, we can procede to describe our ideas and the design choices. In order to do so, we begin with a flowchart that will help us visualize the steps we followed.

	+----------------------------+
	| Load & EDA                 |
	| Data (Scanned documents)   |
	+----------------------------+
            |
            v
	+----------------------------+
	| Feature Extraction         |
	| (LBP, PCA)                 |
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
	| Models                     |
	| (ANN, Autoencoder, VAE)    |
	+----------------------------+
            |
            v
	+----------------------------+
	| Model Evaluation           |
	| (Metrics & Validation)     |
	+----------------------------+


#### Preprocessing choices

In order to properly prepare the data for modelling, some preprocessing steps were taken. 
First, all the images in the dataset were resized to 750x1000. In fact, the EDA showed an average width of 750 and a fixed height of 1000 for all the images. Other dimensions weren't chosen (64x64, 246x246 and so on) because they would lose too much information and since we're dealing with images containing text, it is pivotal to have high quality images.
The images were then converted into numpy arrays to optimize the operation both on a speed and an efficiency level.
Pixels were then normalized in the range [0,1] to help in stabilizing and accelerating the training process. Keeping the pixels to their original rage [0, 255] would have led to a slower optimization of the gradient descent.



#### Feature Extraction

For what concerns feature extraction, we adopted **LBP**, Local Binary Pattern, since it is a texture descriptor that captures the local structure around each pixel. This characteristic makes it perfect for distinguishing different types of scanned documents, such as resumes. Moreover, since we have to deal with a large dataset, LBP was the right choice because it is very computationally efficient ad simple to implement. 
Other options we took into consideration were HOG (Histogram of Oriented Gradients) or SIFT. We tried to adopt them but they were too complex and computationally expensive. 
At this point, the dataset was split into training and test sets.
However, it's worth to mention that LBP features can be noisy and redundant. This is why Principal Component Analysis was applied to reduce dimensionality and simplify the dataset while retaining most of the variance. 
Another option would have been LDA (Linear Discriminant Analysis), but it requires labeled data for finding the linea combinations of features that separate classes, which is not compatible with our task.



#### Algorithms
We adopted a k-means clustering to partition the dataset into 4 distinct and non overlapping clusters, since it is a simple algorithm and minimizes the variance within each cluster. 
We took into consideration DBSCAN and Hierarchical Clustering. As for DBSCAN, we'll discuss the results we obtained in the next section and why we excluded it. Hierarchical clustering was on the other hasnd too computationally expensive and not as scalable to large datasets as k-means.

We then applied the k-means clustering to an ANN, since ANNs are highly flexible and capable of learning complex patterns in the data. They can effectively handle the features extracted by PCA though the use of dense layers and dropout for regularization.
Support Vector Machines and Random Forests were alternatives we considered. However, SVMs may not scale well to large datasets like oura and Random Forests cannot learn patterns as complex as the ones ANNs can learn


- **Artificial Neural Networks (ANN)**: A type of deep learning model that is particularly effective for tasks involving image data, used here for supervised learning to classify résumés.

#### Training Overview
- **Data Normalization**: Adjusting the values in the dataset to a common scale, typically ranging from 0 to 1, without distorting differences in the ranges of values.
- **Data Augmentation**: Techniques such as rotations, zooming, and horizontal flips applied to training data to increase the diversity of data available for training models, thereby improving the generalization ability of the models.
- **Hyperparameter Tuning**: The process of optimizing the parameters that govern the training process of machine learning models, such as learning rate, number of layers, and batch size.








