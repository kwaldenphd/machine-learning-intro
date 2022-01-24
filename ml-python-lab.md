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
- [Next Steps](#next-steps)

# Getting Started With `scikit-learn`

1. The first step is to make sure we have all the necessary packages installed in our Python environment:
- [`NumPy`](https://numpy.org/install/)
- [`SciPy`](https://www.scipy.org/install.html)
- [`matplotlib`](https://matplotlib.org/3.3.3/users/installing.html)
- [`pandas`](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)

2. A couple of options for installing the needed packages.

  * 2a. We can install at the command line using `pip`:
    * `pip install PACKAGE NAME`

  * 2b. We can install using `conda`:
    * `conda install PACKAGE NAME`

  * 2c. To install in a Jupyter notebook environment:
```Python
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install <PACKAGE NAME>
```

3. The links above send you directly to the package installation instructions.

4. To install `scikit-learn`:
- (using pip) `pip install -U scikit-learn`
- (using conda) `conda install -c conda-forge scikit-learn`

5. For a Jupyter notebook environment:
```Python
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install --user numpy
!{sys.executable} -m pip install --user pandas
!{sys.executable} -m pip install --user scipy
!{sys.executable} -m pip install --user matplotlib
!{sys.executable} -m pip install --user sckikit-learn
```

3. The links above send you directly to the package installation instructions.

4. To install `scikit-learn`:
- (using pip) `pip install -U scikit-learn`
- (using conda) `conda install -c conda-forge scikit-learn`

5. For a Jupyter notebook environment:
```Python
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install --user sckikit-learn
```

# Classifying Iris Species

6. This portion of the lab will walk through the process of building a machine leraning application and model to determine petal and sepal length and width measurements for iris flowers.

<p align="center"><a href="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Figure_2_Iris.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Figure_2_Iris.png?raw=true" height="350" /></a></p>

7. In this scenario, we have existing measurements for three iris species.

8. Our goal is to build a machine learning model from the existing measurements to be able to predict the species for a new iris based on its measurements.

9. Because we have labeled measurements as our input data, this is going to be a ***supervised*** machine learning problem.

10. Because we want to predict class membership, this is a ***classification*** problem.

11. The desired output for this model is a single data point--the flower species.

## Meet the Data

12. The iris data is included in `sckikit-learn`.

13. We will load the data by calling the `load_iris` function.

```Python
# import function
from sklearn.datasets import load_iris

# load dataset
iris_dataset = load_iris()
```

14. The `iris_dataset` object is similar to a dictionary, containing keys and values.

15. To see the list of key-value pairs:
```Python
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
```

16. The value for the `DESCR` key provides a short description this dataset.

17. A great place to start in making sense of what data we have.

```Python
print(iris_dataset['DESCR'[:193] + "\n...")
```

18. We can also look at the value for the `target_names` key to see the three species names that are working as classes in the classification algorithm.

```Python
print("Target names:", iris_dataset['target_names'])
```

19. This output tells us that the three iris species we are working with are setosa, versicolor, and virginica.

20. We can access the value for the `feature_names` key to learn more about each feature or data point.

```Python
print("Feature names:\n", iris_dataset['feature_names'])
```

21. We can also see how Python has stored feature data by using `type()`.

```Python
print("Type of data:", type(iris_dataset['data']))
```

22. We can see that the iris data is stored as a `NumPy` array. 

23. Each row in the array corresponds to a flower, and the columns are the four measurements taken for each flower.

24. We can also get a sense of the scope or size of the dataset using `.shape`.

```Python
print("Shape of data:", iris_dataset['data'].shape)
```

25. This output tells us we have measurements for 150 different flowers. 

26. These individual items are called ***samples***. 

27. The sample properties (in our case flower measurements) are called ***features***.

28. The ***shape*** of the `NumPy` array is the number of samples multiplied by the number of features.

29. We can also access feature values for the first five samples.

```Python
print("First five rows of data:\n", iris_dataset['data'][:5])
```

30. We can dig further into how Python has stored this data.

```Python
print("Type of target:", type(iris_dataset['target']))
```

31. The `target` array contains the species for flowers measured.

32. We can also determine the shape of the `target` array.

```Python
print("Shape of target:", iris_dataset['target'].shape)
```

33. Last but not least, we might want to know how species names are represent or encoded in this dataset.

```Python
print("Target:\n", iris_dataset['target'])
```

34. Remember the `iris['target_names']` array tells us how these numbers relate to species name.
- `0`, setosa
- `1`, versicolor
- `2`, virginica

## Training and Testing Data

35. Remember our original problem- we want to use this data to build a machine learning model that can take a new set of measurements and predict the iris species.

36. The first step is to test if the model will actually work before applying it to new measurements.

37. The model remembers the provided training data, so evaluating the model using the training data is not a measure of effectiveness.

38. One option is to split the labeled data into two parts.

39. One part will be the input data for the model, or the training data.

40. The other part will be used to evaluate the model's effectivness. 

41. This second part is called the test data, test set, or hold-out set.

42. Thankfully, `scikit-learn` contains the function `train_test_split` to facilitate the splitting process.

43. The `train_test_split` function uses 75% of the labeled data as training data and reserves 25% as the test set.

44. You can also customize that ratio when using the function.

45. A couple of other notes on `scikit-learn` syntax before we jump back into code.

46. Data is usually denoted with `X` (uppercase x), and labels are denoted by `y` (lowercase y).

47. That nomenclature is based on the mathematical notation `f(X)=y`.

48. `X` is the input, and `y` is the output.

49. `X` is capitalized because the input data is as two-dimensional array (a matrix or table), while the `y` target is a one-dimensional array (a vector).

50. So let's call `train_test_split` on our data.

```Python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
```

51. When called, the function shuffles the dataset using a random number generator. 

52. This is important because remember our original dataset was sorted by label.

53. Without shuffling, we would not have data from all classes in our test set.

54. We can fix the random number generator using the `random_state` parameter. 

55. The `train_test_split` function output is three `NumPy` arrays.

56. The `X_train` array contains 75% of the dataset rows, which will serve as the training data.

57. The `y_train` array contains the labels for `X_train`.

58. The `X_test` array contains the remaining 25%.

59. We can see the format and shape for those output arrays.

```Python
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
```

```Python
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
```

## Inspect the Data

60. Before jumping into building the model, we need to take a look at the data to make sure our question is something that can be answered using machine learning.

61. We also need to know to what degree our data will contain or be able to provide the information we need.

62. Inspecting the data before building the model can also help identify outliers or other inconsistencies.

63. An easy way to inspect data is through preliminary or exploratory visualization.

64. We will do this for the iris data using a ***scatter plot***.

65. A scatter plot will put one feature on the `x` axis, another on the `y` axis, and draw a dot for each data point.

66. Plotting all of our features at once will generate a rather chaotic scatter plot.

67. So we're going to use a ***pair plot*** which creates sub plots for all possible pairs of features.

68. A pair plot is a reasonable solution because we are only dealing with four features, which limits the possible combinations.

69. Keep in mind a pair plot breaks out the unique pairs, so it does not show all features in the same plot.

70. In the plot generated by the code below, data points are colored based on the species they belong to.

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

71. There's a lot more we could get into with `matplotlib` syntax and `pandas`.

72. To learn more: 
  * [Introduction to Matplotlib](https://github.com/kwaldenphd/matplotlib-intro)
  * [More With Matplotlib](https://github.com/kwaldenphd/more-with-matplotlib/)

73. From this pair plot, we can see the three classes separate fairly well based on sepal and petal measurements.

74. This distribution suggests a machine learning model would be able to learn to distinguish species type (or class) based on these measurements.

## Build Your First Model: k-Nearest Neighbors

75. For our first model, we're going to use the k-nearest neighbors classification algorithm.

76. To build this model, we only have to store the training dataset.

77. When making predictions for new data points, the algorithm will find the point in the training data that is closest to the new point.

78. The algorithm then assigns the closest point training data label to the new point.

79. [Click here](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) to learn more about the k-nearest neighbors classification algorithm.

80. The *k* in the algorithm name means that we can consider any number (k) of neighbors when making predictions.

81. For this example, we'll only use a single neighbor.

82. Let's start!

```Python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
```

83. The `knn` object contains the algorithm we will use to build the model, using the training data.

84. It also contains the algorithm that will make predictions for new data points.

85. And third, it contains the information the algorithm has extracted (or learned) from the training data. 

86. A quick note on Python syntax-- all `scikit-learn` machine learning models are implemented in their own classes (called `Estimator` classes).

87. Using a model requires first instantiating the class into an object. 

88. This process also lets us set the model parameters.

89. Now we can use the training data to build the model.

90. The `fit` method (part of the `knn` object) takes arguments that contain our training data.

91. Remember our output from the `train_test_split` function output was three `NumPy` arrays.

92. We will pass `X_train` and `y_train` as arguments to `fit`.

```Python
knn.fit(X_train, y_train)
```

93. The `fit` method returns the `knn` object (modified in place).

94. We are seeing a string representation of our classifier.

95. That representation also shows us which parameters were used to create the model.

96. In this example, we're using the default parameters.

97. There's a lot more we could get into with these parameters, but again we're focusing on the basics in this lab.

## Making Predictions

98. Now that we have a model, we can start to make predictions for unlabeled data.

99. In this example, our unlabeled data would be a series of measurements.

100. Let's say we have an unclassified iris with the following measurements:
  * 5 cm sepal length
  * 2.9 cm sepal width
  * 1 cm petal length
  * 0.2 cm petal width

101. We could store these measurements in a `NumPy` array and calculate its shape (# samples X # of features).

```Python
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)
```

102. Okay- now we have a single flower measurements as a two-dimensional `NumPy` array.

103. We have to go through the process of creating the array, becaues `scikit-learn` requires two-dimensional arrays.

104. Now, we are going to call the `predict` method for the `knn` object to make prediction for `X_new`.

```Python
prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       iris_dataset['target_names'][prediction])
```

105. Our model predicts the `X_new` iris belongs to class `0`, which corresponds to species `setosa`.

## Evaluating the Model

106. But we just gave the model random measurements. 

107. We don't actually know if this prediction is correct.

108. Behold the value of the test set created earlier in this process.

109. The test set gives us data that is labeled but was not used to build the model.

110. We can use the test set data to make a prediction and then compare it to the known label.

111. This process allows us to compute the model accuracy.

112. We'll start by creating label predictions for the test set data contained in `X_test`.

```Python
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)
```

113. Then we can compare `y_pred` to `y_test` to calculate an accuracy score.

```Python
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
```

114. We could also perform this computation using the `knn` object's `score` method.

```Python
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
```

115. This score tells us that, on the test set, the model made the correct species prediction 97% of the time.

116. Extrapolating that accuracy score, we can expect our model to be correct 97% of the time.

<p align="center"><a href="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Fig_2_Snoopy.gif?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Fig_2_Snoopy.gif?raw=true" /></a></p>

117. Congrats- you've built a machine learning model!

# Next Steps

118. [Click here](https://github.com/kwaldenphd/machine-learning-intro#summary) to navigate to the last section of the lab.
