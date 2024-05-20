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

The steps we followed are: 

1) Exploratory Data Analysis, to first inspect the dataset. Since our dataset is made of images, the features analyzed were widht, height, file size, contrast and brightness. This phase is pivotal for all the next steps of the project.
2) Preprocessing, to prepare the data for modelling. The actions taken were resizing, pixel normalization, conversion from images to numpy array, extraction of LBP features and splitting into train and test sets. It was followed by a principal component analysis (PCA) to further reduce the dimensionality of the features.
3) Modelling, to classify the resumes into four classes. The chosen models are: **k-means**, **ANN**, **Autoencoder** and **Variational Autoencoder (VAE)**. We performed extensive validation through cross-validation and analysis of learning curves to ensure the robustness and generalizability of the models.



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
+ First, all the images in the dataset were resized to 750x1000.
   + In fact, the EDA showed an average width of 750 and a fixed height of 1000 for all the images. Other dimensions weren't chosen (64x64, 246x246 and so on) because they would lose too much information and since we're dealing with images containing text, it is pivotal to have high quality images.
+ The images were then converted into numpy arrays to optimize the operation both on a speed and an efficiency level.
+ Pixels were then normalized in the range [0,1] to help in stabilizing and accelerating the training process.
  	+ Keeping the pixels to their original rage [0, 255] would have led to a slower optimization of the gradient descent.



#### Feature Extraction

1) For what concerns feature extraction, we adopted **LBP**, Local Binary Pattern, a texture descriptor that captures the local structure around each pixel.
   - This characteristic makes it perfect for distinguishing different types of scanned documents, such as resumes.
   - Moreover, since we have to deal with a large dataset, LBP was the right choice because it is very computationally efficient ad simple to implement.
   - Other options we took into consideration were HOG (Histogram of Oriented Gradients) or SIFT. We tried to adopt them but they were too complex and computationally expensive. 
2) At this point, the dataset was split into training and test sets.
3) However, it's worth to mention that LBP features can redundant. This is why Principal Component Analysis was applied to reduce dimensionality and simplify the dataset while retaining most of the variance.
   - Another option would have been LDA (Linear Discriminant Analysis), but it requires labeled data for finding the linear combinations of features that separate classes, which is not compatible with our task.
4) We moved to more advanced models, such as autoencoders and Variational Autoencoder.
   - Autoencoders are neural networks designed to learn a compressed representation of the input data. By training an autoencoder to reconstruct the input data from a lower-dimensional encoding, we can capture important features and reduce dimensionality.
   - They are more flexible than PCA and can capture non-linear relationships, which is a big limit of PCA.
5) We also applied a variational autoencoder (VAE) to capture complex, non-linear relationships in the data. This way, meaningful and compact feature representatios are generated, improving clustering and classification performance.
   - While autoencoders can also learn a lower-dimensional representation, VAEs were chosen for their ability to enforce a continuous and smooth latent space, which is beneficial for clustering tasks.




#### Algorithms
- We applied a k-means clustering to both the LBP, PCA, autoencoder and VAE features to partition the dataset into 4 distinct and non overlapping clusters, since it is a simple yet effective algorithm that minimizes the variance within each cluster. It is also easy to implement and computationally efficient.
- We took into consideration DBSCAN and Hierarchical Clustering. As for DBSCAN, the results obtained by its application were mediocre and not as good as k-means.
	- Hierarchical clustering was on the other hand too computationally expensive and not as scalable to large datasets as k-means.

- We then trained an ANN, first to PCA features and then to VAE features, since ANNs are highly flexible and capable of learning complex patterns in the data. They can effectively handle the features extracted by PCA though the use of dense layers and dropout for regularization.
	- LBP was not trained on the ANN since PCA performed better.
	- Support Vector Machines and Random Forests were alternatives we considered. However, SVMs may not scale well to large datasets like ours and Random Forests cannot learn patterns as complex as the ones ANNs can learn.




## Experimental Design

In this section, we describe the experiments conducted to demonstrate and validate the contributions of our project. We focused on comparing different feature extraction methods and clustering algorithms, as well as evaluating the performance of supervised learning models. 

#### Experiment 1: PCA-based Clustering and Classification

- **Purpose**: To evaluate the performance of traditional PCA-based feature extraction for clustering and classification tasks.
  
- **Baseline**: K-Means clustering on PCA-reduced features and classification using an Artificial Neural Network (ANN) trained on these features.
  
- **Evaluation Metrics**:
  - **Clustering Metrics**:
    - **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters. A higher score indicates better-defined clusters.
    - **Davies-Bouldin Index**: Represents the average similarity ratio of each cluster with its most similar cluster. Lower values indicate better clustering.
    - **Calinski-Harabasz Index**: Ratio of the sum of between-cluster dispersion and of within-cluster dispersion. Higher values indicate better clustering.
  - **Classification Metrics**:
    - **Accuracy**: The ratio of correctly predicted instances to the total instances.
    - **Confusion Matrix**: Provides insights into the true positive, true negative, false positive, and false negative predictions.

#### Experiment 2: Autoencoder-based Clustering

- **Purpose**: To assess the effectiveness of traditional autoencoders in learning meaningful features for clustering.
  
- **Baseline**: K-Means clustering on features extracted by the autoencoder.
  
- **Evaluation Metrics**:
  - **Clustering Metrics**: Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.


#### Experiment 3: VAE-based Clustering and Classification

- **Purpose**: To evaluate the performance of Variational Autoencoders (VAEs) in learning compact and meaningful latent representations for clustering and classification.
  
- **Baseline**: K-Means clustering on VAE latent features and classification using an ANN trained on these features.
  
- **Evaluation Metrics**:
  - **Clustering Metrics**: Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index (same as for PCA and autoencoder-based features).
  - **Classification Metrics**: Accuracy, Confusion Matrix.

#### Experiment 4: Comparing Different Clustering Algorithms

- **Purpose**: To compare the effectiveness of different clustering algorithms, specifically K-Means and DBSCAN, on the extracted features.
  
- **Baseline**: K-Means clustering on PCA, autoencoder, and VAE features.
  
- **Evaluation Metrics**:
  - **Clustering Metrics**: Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
  - **Observation**: DBSCAN was found to be less effective than K-Means for our dataset, as it could not form well-defined clusters and was sensitive to the choice of epsilon and min_samples parameters.

#### Experiment 5: Evaluating OCR Performance

- **Purpose**: To evaluate the precision of Optical Character Recognition (OCR) in extracting text from images for potential use in feature extraction.
  
- **Baseline**: Comparison with manual annotations of text.
  
- **Evaluation Metrics**:
  - **Text Recognition Accuracy**: The ratio of correctly recognized characters to the total characters.
  - **Observation**: OCR was not precise and struggled to recognize crucial text elements, making it unsuitable for reliable feature extraction in our context.





## Results


#### Main Findings

1. **PCA-based Clustering and Classification**
   - **Clustering Performance**: PCA with K-Means clustering provided moderately well-defined clusters. However, the clusters were not as distinct as those obtained with advanced models.
     - **Silhouette Score**: 0.45
     - **Davies-Bouldin Index**: 0.96
     - **Calinski-Harabasz Index**: 2344
   - **Classification Performance**: The ANN trained on PCA features achieved the highest accuracy but showed limitations in capturing complex patterns.
     - **Accuracy**: 96%
    

2. **Autoencoder-based Clustering**
   - **Clustering Performance**: The autoencoder improved the clustering quality compared to PCA, indicating that non-linear feature extraction is beneficial.
     - **Silhouette Score**: 0.43
   


3. **VAE-based Clustering and Classification**
   - **Clustering Performance**: VAEs provided the best clustering results, with well-separated clusters, indicating the efficacy of probabilistic latent space representations.
     - **Silhouette Score**: 0.53
     - **Davies-Bouldin Index**: 0.58
     - **Calinski-Harabasz Index**: 3993
   - **Classification Performance**: The ANN trained on VAE features achieved reasonable accuracy but demonstrated the effectiveness of VAEs in capturing complex data distributions.
     - **Accuracy**: 77.86%
  

4. **Comparison of Clustering Algorithms**
   - **K-Means vs. DBSCAN**: K-Means consistently outperformed DBSCAN across all feature sets. DBSCAN's performance was hampered by its sensitivity to parameter selection and inability to handle high-dimensional data effectively.
     - **DBSCAN Clustering Metrics**: Lower silhouette scores and higher Davies-Bouldin indices compared to K-Means, indicating poorer clustering quality.

5. **OCR Performance**
   - **Text Recognition**: The OCR system struggled to accurately recognize crucial text elements, resulting in low precision. This made OCR unsuitable for reliable feature extraction in this context.
     - **Conclusion**: Due to its low accuracy, OCR was not used for further feature extraction and analysis.

#### Figures and Tables


**Figure 1: t-SNE Visualization of Encoded-features-based Clusters**

![t-SNE Visualization of Clusters](images/t-sne_visualization_of_clusters.png)

This t-SNE visualization indicates a well-structured latent space and significant features contributing to the clustering. The clustering seems to be effectively capturing the underlying patterns in the data.


**Figure 2: Confusion Matrix for PCA-based  ANN Classification**

![Confusion Matrix](images/confusion_matrix.png)

The confusion matrix shows that the model performs exceptionally well on the test set. Most classes have very few misclassifications, with the majority of the predicted labels matching the true labels. For instance, Class 0 has a few misclassifications (8 samples predicted as other classes), but Classes 1, 2, and 3 have very high accuracy, with almost no misclassifications.




**Table 1: Clustering Metrics Comparison**

| Feature Set       | Clustering Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|-------------------|----------------------|------------------|----------------------|-------------------------|
| PCA               | K-Means              | 0.45             | 0.96                 | 2344                    |
| Autoencoder       | K-Means              | 0.43             |                      |                         |
| VAE               | K-Means              | 0.53             | 0.58                 | 3993                    |
| PCA               | DBSCAN               | 0.42             | 1.55                 | 31.37                   |
| LBP               | DBSCAN               | -0.25            | 2.59                 | 182.1                   |


**Table 2: Classification Metrics Comparison on Test Data**

| Feature Set       | Classification Model | Accuracy    |
|-------------------|----------------------|-------------|
| PCA               | ANN                  | 97.77%      |
| VAE               | ANN                  | 77.86%      |

These results illustrate the varying strengths of different feature extraction methods. While Principal Component Analysis (PCA) demonstrated superior performance in classification tasks with a high accuracy of 96.84%, advanced models like Variational Autoencoders (VAEs) provided significant improvements in clustering tasks. VAEs, in particular, excelled at capturing complex, non-linear data structures, validating their effectiveness for tasks that require a nuanced understanding of data distributions.



## Conclusions

#### Summary


Our study aimed to compare the effectiveness of different feature extraction methods, specifically Local Binary Patterns (LBP), Principal Component Analysis (PCA), and Variational Autoencoders (VAEs), and the impact of PCA and VAE on clustering and classification tasks using an Artificial Neural Network (ANN) model. 
The results provided insights into the strengths and limitations of each method in different contexts.


PCA-based Model:

The ANN model trained on PCA features achieved a high classification accuracy of 96.84%. This demonstrates PCA's effectiveness in linearly reducing the dimensionality of data while retaining most of the variance. PCA's ability to simplify the data without losing critical information makes it highly suitable for tasks requiring high classification accuracy. However, PCA's limitation lies in its assumption of linearity, which may not capture more complex, non-linear relationships in the data. 

VAE-based Model:

The ANN model trained on VAE features achieved a classification accuracy of 77.86%. It has to be considered though, that VAEs can capture complex and non-linear relationships within the data, which represents a limit of PCAs. Therefore, while VAE features did not outperform PCA in classification accuracy, they can still be valuable for tasks that require capturing more complex data distributions.


In summary, the PCA-based Model achieved the highest accuracy. Moreover, by displaying samples from each cluster we can visually observe a clear distinction between clusters and similarity in layout characteristics within each cluster. 


#### Future Work and Open Questions

While our study highlights the effectiveness of PCA, several questions remain unanswered and present opportunities for future research. One such question is the scalability of VAEs and their performance on much larger datasets. Additionally, the impact of different network architectures and hyperparameters on the performance of VAEs warrants further investigation. Future work could explore the integration of other advanced models, such as Generative Adversarial Networks (GANs), and the application of semi-supervised learning techniques to further enhance feature extraction and classification performance. Moreover, improving the precision of OCR systems and integrating text-based features could provide a more comprehensive approach to document analysis and classification.





