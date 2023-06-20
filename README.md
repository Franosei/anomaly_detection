# anomaly_detection

This PHM Data (https://www.kaggle.com/datasets/hetarthchopra/gearbox-fault-detection-dataset-phm-2009-nasa)  is focused on fault detection and magnitude estimation for a generic gearbox using accelerometer data and information about bearing geometry. 

In this project, I utilized Scala for implementing the K-means algorithm, and R for data visualization purposes. The project focused on anomaly detection in the gearbox fault detection dataset from the NASA PHM 2009 challenge, which is available on Kaggle.

To begin, I imported the dataset into Scala and performed any necessary data preprocessing steps. This involved handling missing values, normalizing the sensor readings, and selecting relevant features for analysis.

Next, I implemented the K-means algorithm using Scala's libraries or custom code. K-means is an unsupervised machine learning algorithm that aims to partition the data into clusters based on their similarity. By iteratively assigning data points to the nearest cluster centroid and updating the centroids, K-means forms clusters that represent distinct patterns in the data.

Once the clustering was performed, I extracted the cluster assignments for each data point. By comparing the distances between data points and their assigned cluster centroids, I identified anomalies as data points that were significantly different from the majority of the data.

For data visualization, I utilized R, a popular language for statistical computing and graphics. I used R's libraries, such as ggplot2 or plotly, to create visualizations that showcased the distribution of the data, the cluster assignments, and any identified anomalies. These visualizations provided insights into the characteristics of the dataset and highlighted any abnormal patterns or outliers.
