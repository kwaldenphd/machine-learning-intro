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









# Project Prompts

# Lab Notebook Questions
