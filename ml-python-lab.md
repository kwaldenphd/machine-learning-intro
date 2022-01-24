# Getting Started With Machine Learning in Python: Lab

<a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license"><img style="border-width: 0;" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" alt="Creative Commons License" /></a>
This tutorial is licensed under a <a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

## Acknowledgements

This lab is based on Chapter 1 "Introduction" from Andreas C. Müller and Sarah Guide, *[Introduction to Machine learning With Python: A Guide for Data Scientists](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/)* (O'Reilly, 2017).

# Table of Contents

- [Overview](#overview)
- [Getting Started With `scikit-learn`](#getting-started-with-scikit-learn)
- [Classifying Iris Species](#classifying-iris-species)
  * [Meet the Data](#meet-the-data)
  * [Training and Testing Data](#training-and-testing-data)
  * [Inspect the Data](#inspect-the-data)
  * [Build Your First Model: k-Nearest Neighbors](#build-your-first-model-k-nearest-neighbors)
  * [Making Predictions](#making-predictions)
  * [Evaluating the Model](#evaluating-the-model)
- [Summary](#summary)
- [Additional Resources](#additional-resources)
- [Lab Notebook Questions](#lab-notebook-questions)

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

<p align="center"><a href="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Figure_2_Iris.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Figure_2_Iris.png?raw=true" height="350" /></a></p>

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

This output tells us that the three iris species we are working with are setosa, versicolor, and virginica.

We can access the value for the `feature_names` key to learn more about each feature or data point.

```Python
print("Feature names:\n", iris_dataset['feature_names'])
```

We can also see how Python has stored feature data by using `type()`.

```Python
print("Type of data:", type(iris_dataset['data']))
```

We can see that the iris data is stored as a `NumPy` array. 

Each row in the array corresponds to a flower, and the columns are the four measurements taken for each flower.

We can also get a sense of the scope or size of the dataset using `.shape`.

```Python
print("Shape of data:", iris_dataset['data'].shape)
```

This output tells us we have measurements for 150 different flowers. 

These individual items are called ***samples***. 

The sample properties (in our case flower measurements) are called ***features***.

The ***shape*** of the `NumPy` array is the number of samples multiplied by the number of features.

We can also access feature values for the first five samples.

```Python
print("First five rows of data:\n", iris_dataset['data'][:5])
```

We can dig further into how Python has stored this data.

```Python
print("Type of target:", type(iris_dataset['target']))
```

The `target` array contains the species for flowers measured.

We can also determine the shape of the `target` array.

```Python
print("Shape of target:", iris_dataset['target'].shape)
```

Last but not least, we might want to know how species names are represent or encoded in this dataset.

```Python
print("Target:\n", iris_dataset['target'])
```

Remember the `iris['target_names']` array tells us how these numbers relate to species name.
- `0`, setosa
- `1`, versicolor
- `2`, virginica

## Training and Testing Data

Remember our original problem- we want to use this data to build a machine learning model that can take a new set of measurements and predict the iris species.

The first step is to test if the model will actually work before applying it to new measurements.

The model remembers the provided training data, so evaluating the model using the training data is not a measure of effectiveness.

One option is to split the labeled data into two parts.

One part will be the input data for the model, or the training data.

The other part will be used to evaluate the model's effectivness. 

This second part is called the test data, test set, or hold-out set.

Thankfully, `scikit-learn` contains the function `train_test_split` to facilitate the splitting process.

The `train_test_split` function uses 75% of the labeled data as training data and reserves 25% as the test set.

You can also customize that ratio when using the function.

A couple of other notes on `scikit-learn` syntax before we jump back into code.

Data is usually denoted with `X` (uppercase x), and labels are denoted by `y` (lowercase y).

That nomenclature is based on the mathematical notation `f(x)=y`.

`X` is the input, and `y` is the output.

`X` is capitalized because the input data is as two-dimensional array (a matrix or table), while the `y` target is a one-dimensional array (a vector).

So let's call `train_test_split` on our data.

```Python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
```

When called, the function shuffles the dataset using a random number generator. 

This is important because remember our original dataset was sorted by label.

Without shuffling, we would not have data from all classes in our test set.

We can fix the random number generator using the `random_state` parameter. 

The `train_test_split` function output is three `NumPy` arrays.

The `X_train` array contains 75% of the dataset rows, which will serve as the training data.

The `y_train` array contains the labels for `X_train`.

The `X_test` array contains the remaining 25%.

We can see the format and shape for those output arrays.

```Python
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
```

```Python
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
```

## Inspect the Data

Before jumping into building the model, we need to take a look at the data to make sure our question is something that can be answered using machine learning.

We also need to know to what degree our data will contain or be able to provide the information we need.

Inspecting the data before building the model can also help identify outliers or other inconsistencies.

An easy way to inspect data is through preliminary or exploratory visualization.

We will do this for the iris data using a ***scatter plot***.

A scatter plot will put one feature on the `x` axis, another on the `y` axis, and draw a dot for each data point.

Plotting all of our features at once will generate a rather chaotic scatter plot.

So we're going to use a ***pair plot*** which creates sub plots for all possible pairs of features.

A pair plot is a reasonable solution because we are only dealing with four features, which limits the possible combinations.

Keep in mind a pair plot breaks out the unique pairs, so it does not show all features in the same plot.

In the plot generated by the code below, data points are colored based on the species they belong to.

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

There's a lot more we could get into with `matplotlib` syntax and `pandas`.

To learn more: 
- [Introduction to Matplotlib](https://github.com/kwaldenphd/matplotlib-intro)
- [More With Matplotlib](https://github.com/kwaldenphd/more-with-matplotlib/)

From this pair plot, we can see the three classes separate fairly well based on sepal and petal measurements.

This distribution suggests a machine learning model would be able to learn to distinguish species type (or class) based on these measurements.

## Build Your First Model: k-Nearest Neighbors

For our first model, we're going to use the k-nearest neighbors classification algorithm.

To build this model, we only have to store the training dataset.

When making predictions for new data points, the algorithm will find the point in the training data that is closest to the new point.

The algorithm then assigns the closest point training data label to the new point.

[Click here](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) to learn more about the k-nearest neighbors classification algorithm.

The *k* in the algorithm name means that we can consider any number (k) of neighbors when making predictions.

For this example, we'll only use a single neighbor.

Let's start!

```Python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
```

The `knn` object contains the algorithm we will use to build the model, using the training data.

It also contains the algorithm that will make predictions for new data points.

And third, it contains the information the algorithm has extracted (or learned) from the training data. 

A quick note on Python syntax-- all `scikit-learn` machine learning models are implemented in their own classes (called `Estimator` classes).

Using a model requires first instantiating the class into an object. 

This process also lets us set the model parameters.

Now we can use the training data to build the model.

The `fit` method (part of the `knn` object) takes arguments that contain our training data.

Remember our output from the `train_test_split` function output was three `NumPy` arrays.

We will pass `X_train` and `y_train` as arguments to `fit`.

```Python
knn.fit(X_train, y_train)
```

The `fit` method returns the `knn` object (modified in place).

We are seeing a string representation of our classifier.

That representation also shows us which parameters were used to create the model.

In this example, we're using the default parameters.

There's a lot more we could get into with these parameters, but again we're focusing on the basics in this lab.

## Making Predictions

Now that we have a model, we can start to make predictions for unlabeled data.

In this example, our unlabeled data would be a series of measurements.

Let's say we have an unclassified iris with the following measurements:
- 5 cm sepal length
- 2.9 cm sepal width
- 1 cm petal length
- 0.2 cm petal width

We could store these measurements in a `NumPy` array and calculate its shape (# samples X # of features).

```Python
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)
```

Okay- now we have a single flower measurements as a two-dimensional `NumPy` array.

We have to go through the process of creating the array, becaues `scikit-learn` requires two-dimensional arrays.

Now, we are going to call the `predict` method for the `knn` object to make prediction for `X_new`.

```Python
prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       iris_dataset['target_names'][prediction])
```

Our model predicts the `X_new` iris belongs to class `0`, which corresponds to species `setosa`.

## Evaluating the Model

But we just gave the model random measurements. 

We don't actually know if this prediction is correct.

Behold the value of the test set created earlier in this process.

The test set gives us data that is labeled but was not used to build the model.

We can use the test set data to make a prediction and then compare it to the known label.

This process allows us to compute the model accuracy.

We'll start by creating label predictions for the test set data contained in `X_test`.

```Python
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)
```

Then we can compare `y_pred` to `y_test` to calculate an accuracy score.

```Python
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
```

We could also perform this computation using the `knn` object's `score` method.

```Python
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
```

This score tells us that, on the test set, the model made the correct species prediction 97% of the time.

Extrapolating that accuracy score, we can expect our model to be correct 97% of the time.

<p align="center"><a href="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Fig_2_Snoopy.gif?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Fig_2_Snoopy.gif?raw=true" /></a></p>

Congrats- you've built a machine learning model!

## Summary

This lab covered a lot of ground. Let's recap.

We started by unpacking some of the key terms used in machine learning, including supervised and unsupervised machine learning algorithms.

We also talked about some of the common statistical methods used in those algorithms, including classification and regression.

We then walked through an overview of the core Python libraries and packages used in machine learning, with a focus on the `SciPy` stack.

The last half of the lab focused on building a model that predicts iris species based on sepal and petal measurements.

For our three-class classification problem, the iris species were classes, and the species for an individual flower was its label.

We stored the measurements and labels in NumPy arrays.

Our model used the k-nearest neighbors classification algorithm and split labeled data into a training set and a test set.

We passed those training data parameters to the model.

We then used the test set data to calculate the model's accuracy.

Putting that all together:

```Python
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
```

The core components of this example (`fit`, `predict`, and `score`) are common across other `scikit-learn` supervised algorithms.

# Additional Resources

Later in the semester, you will have the opportunity to build your own machine learning model. 

The six steps outlined above can get you started, but there's obviously additional research needed when making decisions about approaches, methods, and models.

Starting with a method or concept's Wikipedia page can give you the broad strokes as well as links or citatations for additional resources.

Andreas C. Müller and Sarah Guide's accessible *Introduction to Machine learning With Python: A Guide for Data Scientists* (O'Reilly, 2017) is where I would start with additional research. 
- [Publisher website](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/)
- [Link to access via Google Drive](https://drive.google.com/file/d/1VHBuayX6PoZZrFaps-HLs3exXoLPSlSM/view?usp=sharing) (ND users only)
- [Code repository](https://github.com/amueller/introduction_to_ml_with_python)

[Manning Publications](https://www.manning.com/) has a number of titles on machine learning and deep learning that are also valuable starting places.

# Lab Notebook Questions

Q1: Write a narrative that documents and describes your experience working through this lab. What challenges did you face, and how did you solve them? What did you learn about machine learning through this lab? How are you thinking about machine learning differently after this lab? 
- You are welcome (but not required) to include Python code as part of that narrative.
