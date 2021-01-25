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

<p align="center"><a href="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Dilbert_ML.gif?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Dilbert_ML.gif?raw=true" /></a></p>

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

The purpose of this lab is to help demystify the broad strokes of how machine learning works, and provide an introduction to machine learning in the Python programming environment. 

# Machine Learning in Python

Python provides a range of options for machine learning work.

The example in this lab uses `scikit-learn`, but there are other options.

## Core Libraries

Nearly all Python machine learning libraries are built on (or integrate) a few core libraries.

### `NumPy`

According to [package documentation](https://numpy.org/doc/stable/user/whatisnumpy.html), "`NumPy` is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more."

In `NumPy`, data are stored as list-like objects called arrays.

`NumPy` allows users to access, split, reshape, join, etc. data stored in arrays.

### `pandas`

As described in [package documentation](https://pandas.pydata.org/), `pandas` is "a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language."

`pandas` enables Python to understand or interact with structured data as structured data.

Software developers at AQR Capital Management began working on a Python-based tool (written in a combination of C and Python) for quantitative data analysis in 2008.

The initial open-source version of `pandas` was released in 2008.

At its core, "pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series" ([Wikipedia](https://en.wikipedia.org/wiki/Pandas_(software))).

The name `pandas` is derived from "panel data," an econometrics term used to describe particular types of datasets.

The name `pandas` is also a play on "Python data analysis."

For more on the history and origins of `pandas`, check out Wes McKinney's "[`pandas`: a Foundational Python Library for Data Analysis and Statistics](https://www.dlr.de/sc/Portaldata/15/Resources/dokumente/pyhpc2011/submissions/pyhpc2011_submission_9.pdf)" 2011 paper.

### `matplotlib`

For our purposes, a plot is defined as "a graphic representation (such as a chart)" (Merriam Webster).

These graphic representations of data are often called charts, graphs, figures, etc.

In the context of programming, computer science, and data science, we refer to these as plots.

We can generate plots for data stored in pandas using the `matplotlib` package.

`matplotlib` was developed in 2002 as a MATLAB-like plotting interface for Python.

"Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python...Matplotlib produces publication-quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shell, web application servers, and various graphical user interface toolkits" ([Matplotlib documentation, Github](https://github.com/matplotlib/matplotlib))

As described by the original developer John Hunter, "Matplotlib is a library for making 2D plots of arrays in Python. Although it has its origins in emulating the MATLAB graphics commands, it is independent of MATLAB, and can be used in a Pythonic, object oriented way. Although Matplotlib is written primarily in pure Python, it makes heavy use of NumPy and other extension code to provide good performance even for large arrays. Matplotlib is designed with the philosophy that you should be able to create simple plots with just a few commands, or just one! If you want to see a histogram of your data, you shouldn't need to instantiate objects, call methods, set properties, and so on; it should just work."

For more on `matplotlib`'s development and history: John Hunter, "[History](https://matplotlib.org/users/history.html)" Matplotlib (2008)

### `SciPy`

"SciPy is a free and open-source Python library used for scientific computing and technical computing. SciPy contains modules for optimization, linear algebra, integration, interpolation, special functions, FFT, signal and image processing, ODE solvers and other tasks common in science and engineering. SciPy builds on the NumPy array object and is part of the NumPy stack which includes tools like Matplotlib, pandas and SymPy, and an expanding set of scientific computing libraries. This NumPy stack has similar users to other applications such as MATLAB, GNU Octave, and Scilab. The NumPy stack is also sometimes referred to as the SciPy stack" ([Wikipedia](https://en.wikipedia.org/wiki/SciPy)).

For more on `SciPy`, consult the [package documentation](https://www.scipy.org/scipylib/index.html).

## Machine Learning Libraries

So putting that all together, we have the following packages as the the foundation for our machine learning environment:
- `NumPy`
- `SciPy`
- `matplotlib`
- `pandas`

This combination of packages is part of what is known as the [`SciPy` stack](https://www.scipy.org/stackspec.html).

### Scikit-learn

"Scikit-learn...is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy" ([Wikipedia](https://en.wikipedia.org/wiki/Scikit-learn)).

For more on `scikit-learn`: https://scikit-learn.org

`scikit-learn` supports a wide range of supervised and unsupervised machine learning algorithms.

### Deep Learning Packages

#### TensorFlow

Developed by the team at Google Brain, "TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications" (["Why TensorFlow"](https://www.tensorflow.org/).

#### Keras

"Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation" (["About Keras"](https://keras.io/about/).

#### PyTorch

"PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab (FAIR). It is free and open-source software released under the Modified BSD license...A number of pieces of Deep Learning software are built on top of PyTorch, including Tesla Autopilot [and] Uber's Pyro" ([Wikipedia](https://en.wikipedia.org/wiki/PyTorch)).

For more on PyTorch: https://pytorch.org/

# Getting Started With `scikit-learn`

First step is to make sure we have all the necessary packages installed in our Python environment:
- [`NumPy`](https://numpy.org/install/)
- [`SciPy`](https://www.scipy.org/install.html)
- [`matplotlib`](https://matplotlib.org/3.3.3/users/installing.html)
- [`pandas`](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)

We can install at the command line using `pip`:
- `pip install PACKAGE NAME`

We can install using `conda`:
- `conda install PACKAGE NAME`

To install in a Jupyter notebook environment:
```Python
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install <PACKAGE NAME>
```

The links above send you directly to the package installation instructions.

To install `scikit-learn`:
- (using pip) `pip install -U scikit-learn`
- (using conda) `conda install -c conda-forge scikit-learn`

For a Jupyter notebook environment:
```Python
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install --user sckikit-learn
```

# Classifying Iris Species

This portion of the lab will walk through the process of building a machine leraning application and model to determine petal and sepal length and width measurements for iris flowers.

<p align="center"><a href="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Figure_2_Iris.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Figure_2_Iris.png?raw=true" /></a></p>

In this scenario, we have existing measurements for three iris species.

Our goal is to build a machine learning model from the existing measurements to be able to predict the species for a new iris based on its measurements.

Because we have labeled measurements as our input data, this is going to be a ***supervised*** machine learning problem.

Because we want to predict class membership, this is a ***classification*** problem.

The desired output for this model is a single data point--the flower species.

## Meet the Data

The iris data is included in `sckikit-learn`.

We will load the data by calling the `load_iris` function.

```Python
# import function
from sklearn.datasets import load_iris

# load dataset
iris_dataset = load_iris()
```

The `iris_dataset` object is similar to a dictionary, containing keys and values.

To see the list of key-value pairs:
```Python
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
```

The value for the `DESCR` key provides a short description this dataset.

A great place to start in making sense of what data we have.

```Python
print(iris_dataset['DESCR'[:193] + "\n...")
```

We can also look at the value for the `target_names` key to see the three species names that are working as classes in the classification algorithm.

```Python
print("Target names:", iris_dataset['target_names'])
```

```Python
print("Feature names:\n", iris_dataset['feature_names'])
```

```Python
print("Type of data:", type(iris_dataset['data']))
```

```Python
print("Shape of data:", iris_dataset['data'].shape)
```

```Python
print("First five rows of data:\n", iris_dataset['data'][:5])
```

```Python
print("Type of target:", type(iris_dataset['target']))
```

```Python
print("Shape of target:", iris_dataset['target'].shape)
```

```Python
print("Target:\n", iris_dataset['target'])
```

## Training and Testing Data

```Python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
```

```Python
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
```

```Python
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
```

## What Happened?

```Python
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)
```

<p align="center"><a href="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Figure-3.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Figure_3.png?raw=true" /></a></p>

## Build Your First Model: k-Nearest Neighbors

```Python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
```

```Python
knn.fit(X_train, y_train)
```

## Making Predictions

```Python
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)
```

```Python
prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       iris_dataset['target_names'][prediction])
```

## Evaluating the Model

```Python
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)
```

```Python
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
```

```Python
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
```

## What Now

```Python
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
```

<p align="center"><a href="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Fig_2_Snoopy.gif?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Fig_2_Snoopy.gif?raw=true" /></a></p>

# Additional Resources

Later in the semester, you will have the opportunity to build your own machine learning model. 

The six steps outlined above can get you started, but there's obviously additional research needed when making decisions about approaches, methods, and models.

Starting with a method or concept's Wikipedia page can give you the broad strokes as well as links or citatations for additional resources.

Andreas C. Müller and Sarah Guide's accessible *Introduction to Machine learning With Python: A Guide for Data Scientists* (O'Reilly, 2017) is where I would start with additional research. 
- [Publisher website](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/)
- [Link to access via Google Drive](https://drive.google.com/file/d/1VHBuayX6PoZZrFaps-HLs3exXoLPSlSM/view?usp=sharing) (ND users only)
- [Code repository](https://github.com/amueller/introduction_to_ml_with_python)

[Manning Publications](https://www.manning.com/) has a number of titles on machine learning and deep learning that are also valuable starting places.


# Project Prompts

# Lab Notebook Questions
