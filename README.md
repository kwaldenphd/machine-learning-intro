# Getting Started With Machine Learning in Python

<a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license"><img style="border-width: 0;" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" alt="Creative Commons License" /></a>
This tutorial is licensed under a <a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

## Lab Goals

This lab provides an overview of 

## Acknowledgements

# Table of Contents

# Overview

There are two overarching categories of machine learning algorithms: supervised and unsupervised.

## Supervised

***Supervised machine learning algorithms*** build a model based on data that includes inputs as well as outputs.

This ***training data*** includes example inputs and desired outputs.

A supervised algorithm uses the training data to create (or learn) a function that will predict the output that would be associated with new inputs.

An optimal function will correctly determine the output for inputs not in the original training data. 

There are three primary types of supervised machine learning algorithms.

### Active Learning

In an ***active learning algorithm***, the user interacts with the learning algorithm, labeling new data points as directed by the algorithm.

Active learning algorithms are useful when faced with a large amount of unlabeled data that would benefit from some level of labeling. 

Each iteration of an active learning algorithm divides the data into three subsets.
- Data points with ***known*** labels
- Data points with ***unknown*** labels
- Subset of unknown data points that is ***chosen*** to be labeled. 

The iterative approach limits the number of examples needed for effective supervised learning. 

However, the uninformative examples can limit the algorithm's effectiveness.

### Classification

A ***classification algorithm*** uses training data to determine what category a new observation belongs to.

The training data provided to the algorithm includes observations with known category membership.

Classification algorithms have a limited set of output values.

Part of the work of the algorithm involves analyzing indivdual observations based on quantifiable properties. 

These explanatory variables (or features) can fall into a few main types:
- ***Binary*** (on or off)
- ***Categorical*** (days of the week, months of the year, etc.)
- ***Ordinal*** (based on numerical value or size; large/medium/small)
- ***Integer-valued*** (integer value that represents number of occurances or similar meaning)
- ***Real-valued*** (raw measurement value)

An algorithm that implements classification is called a ***classifier***.

Examples might include an algorithm that classifies unstructured text by part of speech, or an algorithm that classifies nouns as a person, location, or organization.

The model developed by the algorithm is known as a classifier, which determines how the system classifies new inputs or unlabeled data.

### Regression

A ***regression algorithm*** estimates the relationship between a dependent variable (sometimes called the outcome variable) and one or more independent variables (sometimes called predictors, covariates, or features).

In machine learning, regression analysis is used for prediction and forecasting, because it predicts the value of a dependent variable based on the value(s) of an independent variable.

For a linear regression model, the dependent or outcome variable is plotted on the `Y` axis, while the independent or predictor variable(s) are plotted on the `X` axis.

Examples of a linear regression model might include predicting weight based on height, or air pressure based on elevation, etc.

## Unsupervised

***Unsupervised machine learning algorithms*** take input data that has not been labeled, classified, or categorized and find structure in that data, most often through grouping or clustering data points.

On a fundamental level, an unsupervised machine learning algorithm engages in some sort of meaning making, as it attempts to build a compact internal representation of the world contained in the input data.

Unsupervised machine learning falls into two primary methods: principal component analysis (PCA) and cluster analysis.

Other types of unsupervised machine learning include anomaly detection and neural networks.

### Principal Component Analysis

***Principal component analysis*** is used to understand relationships across variables. 

As a statistical technique, PCA reduces the dimensionality of an input dataset by creating new variables (principal components) that are weighted to represent relationships in the input dataset.

For example, when applied to unstructured textual data, PCA will calculate two principal components, terms that are more or less significant in each component, as well as the relationship of the two components.

To see this method in action: 
- Luling Huang, "[Principal Component Analysis: Unsupervised Learning of Textual Data Part III](https://sites.temple.edu/tudsc/2019/03/12/unsupervised-learning-of-textual-data-iii-principal-component-analysis/)" *Temple University Digital Scholarship Center Blog* (12 March 2019).
- Hugh Craig, "[Stylistic Analysis and Authorship Studies](https://sites.temple.edu/tudsc/2019/03/12/unsupervised-learning-of-textual-data-iii-principal-component-analysis/)" in *A Companion to Digital Humanities* (Blackwell, 2004).
- Ted Underwood, "['Plot arcs' in the novel](https://tedunderwood.com/2015/01/03/plot-arcs-in-the-novel/)" *The Stone in the Shell blog* (3 January 2015).
- Ben Schmidt, "[Fundamental plot arcs, seen through multidimensional analysis of thousands of TV and movie scripts](http://sappingattention.blogspot.com/2014/12/fundamental-plot-arcs-seen-through.html)" *Sapping Attention blog* (16 December 2014).

And when in doubt, start with George Dallas, "[Principal Component Analysis 4 Dummies: Eigenvenctors, Eigenvalues and Dimension Reduction](https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/)" *personal blog* (30 October 2013).

### Cluster Analysis

A ***cluster analysis algorithm*** creates groups (or clusters) from a set of objects, with the goal of objects in each group (or cluster) being more similar to each other than the objects in othre groups (clusters).

There are a number of different specific clustering algorithms that employ different methods of calculating and forming clusters.

Some of the common clustering methods include:
- Density-based methods, in which cluster parameters have to do with higher and lower density regions
- Hierarchical-based methods, in which clusters have a tree-like structure
- Partitioning methods, in which objects are partitioned into clusters and each partition forms a cluster
- Grid-based methods, in which a grid-like structure of cells exists in the data space

# Machine Learning Workflow Fundamentals

DILBERT COMIC

<p align="center"><a href="https://github.com/kwaldenphd/machine-learning-intro/blob/main/Figure_1.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/machine-learning-intro/blob/main/Figure_1.png?raw=true" /></a></p>

At this point, your brain is probably hurting. Mine is.

Let's take a step back and consider the fundamentals of a machine learning workflow.

Step 1- identify the problem or question, think about options for answering or solving, and decide if machine learning will be a useful tool.

Step 2- identify and analyze available datasets to see if sufficient data is available to build a model

Step 3- transform the data into a tabular structure so it can be the input for a machine learning model

Step 4- train the model (or more accurately engage in an iterative process of training the model)

Step 5- evaluate the model to see if (and how effectively) it solves the original question/problem

Step 6- deploy the model

As you can imagine, this is an iterative process that takes TIME.

Time to determine the central problem or question, and figure out if machine learning is a good fit.

Time to figure out what data is needed to build a model, identify or collect that data, and wrangle it into a tabular structure.

Time to figure out what type of model is the best fit for the question/problem AS WELL AS the input data.

Time to train the model and evaluate the effectiveness of the model.

Time to figure out what the model has to say about your original question or problem.

In short, TIME.

For non-specialists, the complexity of the math happening underneath a machine learning algorithm can be overwhelming and daunting. 

But our goal here is not to become overnight experts in applied statistics. 

The purpose of this lab is to help demystify the broad strokes of how machine learning works, and 

# Project Prompts

# Lab Notebook Questions
