# Iris Dataset Project

An exploration on the famous Iris data set. Our goal is to predict iris class
by the features given .

Guidance from : https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

## Dataset Analysis
I performed an initial analysis using Jupyter Notebook and discovered that
petal width and petal length clearly separate iris-setosa from the other
two classes. We also saw information that the dataset separated each class
equally.

## Machine learning
We did a comparison of ml models:
* Logistic Regressions
* K-Nearest neighbor
* Decision Tree classifier
* Gaussian Naive Bayes
* Support Vector Classifier

Using the seed recommended by the guide, we determined that an SVC was the most
accurate in our comparisons. After using the SVC by itself, we also determined
that kernel parameters between 'linear' and 'rbf' do not matter. Removing the
random seed from the training set, the accuracy of the kernels were roughly the
same. In the end, we got around 90%+ accuracy on the SVC classifier. 
