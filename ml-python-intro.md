# Getting Started With Machine Learning in Python: Overview

<a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license"><img style="border-width: 0;" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" alt="Creative Commons License" /></a>
This tutorial is licensed under a <a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# Table of Contents

- [Overview](#overview)
  * [Supervised](#supervised)
    * [Active Learning](#active-learning)
    * [Classification](#classification)
    * [Regression](#regression)
  * [Unsupervised](#unsupervised)
    * [Principal Component Analysis](#principal-component-analysis)
    * [Cluster Analysis](#cluster-analysis)
- [Machine Learning Workflow Fundamentals](#machine-learning-workflow-fundamentals)
- [Machine Learning in Python](#machine-learning-in-python)
  * [Core Libraries](#core-libraries)
    * [`NumPy`](#numpy)
    * [`pandas`](#pandas)
    * [`matplotlib`](#matplotlib)
    * [`SciPy`](#scipy)
  * [Machine Learning Libraries](#machine-learning-libraries)
    * [`scikit-learn`](#scikit-learn)
    * [Deep Learning](#deep-learning)
      * [TensorFlow](#tensorflow)
      * [Keras](#keras)
      * [PyTorch](#pytorch)
- [Next Steps](#next-steps)

# Overview

1. There are two overarching categories of machine learning algorithms: supervised and unsupervised.

## Supervised

2. ***Supervised machine learning algorithms*** build a model based on data that includes inputs as well as outputs.

3. This ***training data*** includes example inputs and desired outputs.

4. A supervised algorithm uses the training data to create (or learn) a function that will predict the output that would be associated with new inputs.

5. An optimal function will correctly determine the output for inputs not in the original training data. 

6. There are three primary types of supervised machine learning algorithms.

### Active Learning

7. In an ***active learning algorithm***, the user interacts with the learning algorithm, labeling new data points as directed by the algorithm.

8. Active learning algorithms are useful when faced with a large amount of unlabeled data that would benefit from some level of labeling. 

9. Each iteration of an active learning algorithm divides the data into three subsets.
- Data points with ***known*** labels
- Data points with ***unknown*** labels
- Subset of unknown data points that is ***chosen*** to be labeled. 

10. The iterative approach limits the number of examples needed for effective supervised learning. 

11. However, the uninformative examples can limit the algorithm's effectiveness.

### Classification

12. A ***classification algorithm*** uses training data to determine what category a new observation belongs to.

13. The training data provided to the algorithm includes observations with known category membership.

14. Classification algorithms have a limited set of output values.

15. Part of the work of the algorithm involves analyzing indivdual observations based on quantifiable properties. 

16. These explanatory variables (or features) can fall into a few main types:
- ***Binary*** (on or off)
- ***Categorical*** (days of the week, months of the year, etc.)
- ***Ordinal*** (based on numerical value or size; large/medium/small)
- ***Integer-valued*** (integer value that represents number of occurances or similar meaning)
- ***Real-valued*** (raw measurement value)

17. An algorithm that implements classification is called a ***classifier***.

18. Examples might include an algorithm that classifies unstructured text by part of speech, or an algorithm that classifies nouns as a person, location, or organization.

19. The model developed by the algorithm is known as a classifier, which determines how the system classifies new inputs or unlabeled data.

### Regression

20. A ***regression algorithm*** estimates the relationship between a dependent variable (sometimes called the outcome variable) and one or more independent variables (sometimes called predictors, covariates, or features).

21. In machine learning, regression analysis is used for prediction and forecasting, because it predicts the value of a dependent variable based on the value(s) of an independent variable.

22. For a linear regression model, the dependent or outcome variable is plotted on the `Y` axis, while the independent or predictor variable(s) are plotted on the `X` axis.

23. Examples of a linear regression model might include predicting weight based on height, or air pressure based on elevation, etc.

## Unsupervised

24. ***Unsupervised machine learning algorithms*** take input data that has not been labeled, classified, or categorized and find structure in that data, most often through grouping or clustering data points.

25. On a fundamental level, an unsupervised machine learning algorithm engages in some sort of meaning making, as it attempts to build a compact internal representation of the world contained in the input data.

26. Unsupervised machine learning falls into two primary methods: principal component analysis (PCA) and cluster analysis.

27. Other types of unsupervised machine learning include anomaly detection and neural networks.

### Principal Component Analysis

28. ***Principal component analysis*** is used to understand relationships across variables. 

29. As a statistical technique, PCA reduces the dimensionality of an input dataset by creating new variables (principal components) that are weighted to represent relationships in the input dataset.

30. For example, when applied to unstructured textual data, PCA will calculate two principal components, terms that are more or less significant in each component, as well as the relationship of the two components.

31. To see this method in action: 
- Luling Huang, "[Principal Component Analysis: Unsupervised Learning of Textual Data Part III](https://sites.temple.edu/tudsc/2019/03/12/unsupervised-learning-of-textual-data-iii-principal-component-analysis/)" *Temple University Digital Scholarship Center Blog* (12 March 2019).
- Hugh Craig, "[Stylistic Analysis and Authorship Studies](https://sites.temple.edu/tudsc/2019/03/12/unsupervised-learning-of-textual-data-iii-principal-component-analysis/)" in *A Companion to Digital Humanities* (Blackwell, 2004).
- Ted Underwood, "['Plot arcs' in the novel](https://tedunderwood.com/2015/01/03/plot-arcs-in-the-novel/)" *The Stone in the Shell blog* (3 January 2015).
- Ben Schmidt, "[Fundamental plot arcs, seen through multidimensional analysis of thousands of TV and movie scripts](http://sappingattention.blogspot.com/2014/12/fundamental-plot-arcs-seen-through.html)" *Sapping Attention blog* (16 December 2014).

32. And when in doubt, start with George Dallas, "[Principal Component Analysis 4 Dummies: Eigenvenctors, Eigenvalues and Dimension Reduction](https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/)" *personal blog* (30 October 2013).

### Cluster Analysis

33. A ***cluster analysis algorithm*** creates groups (or clusters) from a set of objects, with the goal of objects in each group (or cluster) being more similar to each other than the objects in other groups (clusters).

34. There are a number of different specific clustering algorithms that employ different methods of calculating and forming clusters.

35. Some of the common clustering methods include:
  * Density-based methods, in which cluster parameters have to do with higher and lower density regions
  * Hierarchical-based methods, in which clusters have a tree-like structure
  * Partitioning methods, in which objects are partitioned into clusters and each partition forms a cluster
  * Grid-based methods, in which a grid-like structure of cells exists in the data space

# Machine Learning Workflow Fundamentals

<p align="center"><a href="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Dilbert_ML.gif?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/machine-learning-intro/blob/main/figures/Dilbert_ML.gif?raw=true" /></a></p>

36. At this point, your brain is probably hurting. Mine is.

37. Let's take a step back and consider the fundamentals of a machine learning workflow.

  * Step 1- identify the problem or question, think about options for answering or solving, and decide if machine learning will be a useful tool.

  * Step 2- identify and analyze available datasets to see if sufficient data is available to build a model

  * Step 3- transform the data into a tabular structure so it can be the input for a machine learning model

  * Step 4- train the model (or more accurately engage in an iterative process of training the model)

  * Step 5- evaluate the model to see if (and how effectively) it solves the original question/problem

  * Step 6- deploy the model

38. As you can imagine, this is an iterative process that takes TIME.

  * Time to determine the central problem or question, and figure out if machine learning is a good fit.

  * Time to figure out what data is needed to build a model, identify or collect that data, and wrangle it into a tabular structure.

  * Time to figure out what type of model is the best fit for the question/problem AS WELL AS the input data.

  * Time to train the model and evaluate the effectiveness of the model.

  * Time to figure out what the model has to say about your original question or problem.

  * In short, TIME.

39. For non-specialists, the complexity of the math happening underneath a machine learning algorithm can be overwhelming and daunting. 

40. But our goal here is not to become overnight experts in applied statistics. 

41. The purpose of this lab is to help demystify the broad strokes of how machine learning works, and provide an introduction to machine learning in the Python programming environment. 

# Machine Learning in Python

42. Python provides a range of options for machine learning work.

43. The example in this lab uses `scikit-learn`, but there are other options.

## Core Libraries

44. Nearly all Python machine learning libraries are built on (or integrate) a few core libraries.

### `NumPy`

45. According to [package documentation](https://numpy.org/doc/stable/user/whatisnumpy.html), "`NumPy` is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more."

46. In `NumPy`, data are stored as list-like objects called arrays.

47. `NumPy` allows users to access, split, reshape, join, etc. data stored in arrays.

### `pandas`

48. As described in [package documentation](https://pandas.pydata.org/), `pandas` is "a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language."

49. `pandas` enables Python to understand or interact with structured data as structured data.

50. Software developers at AQR Capital Management began working on a Python-based tool (written in a combination of C and Python) for quantitative data analysis in 2008.

51. The initial open-source version of `pandas` was released in 2008.

52. At its core, "pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series" ([Wikipedia](https://en.wikipedia.org/wiki/Pandas_(software))).

53. The name `pandas` is derived from "panel data," an econometrics term used to describe particular types of datasets.

54. The name `pandas` is also a play on "Python data analysis."

55. For more on the history and origins of `pandas`, check out Wes McKinney's "[`pandas`: a Foundational Python Library for Data Analysis and Statistics](https://www.dlr.de/sc/Portaldata/15/Resources/dokumente/pyhpc2011/submissions/pyhpc2011_submission_9.pdf)" 2011 paper.

### `matplotlib`

55. For our purposes, a plot is defined as "a graphic representation (such as a chart)" (Merriam Webster).

56. These graphic representations of data are often called charts, graphs, figures, etc.

57. In the context of programming, computer science, and data science, we refer to these as plots.

58. We can generate plots for data stored in pandas using the `matplotlib` package.

59. `matplotlib` was developed in 2002 as a MATLAB-like plotting interface for Python.

60. "Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python...Matplotlib produces publication-quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shell, web application servers, and various graphical user interface toolkits" ([Matplotlib documentation, Github](https://github.com/matplotlib/matplotlib))

61. As described by the original developer John Hunter, "Matplotlib is a library for making 2D plots of arrays in Python. Although it has its origins in emulating the MATLAB graphics commands, it is independent of MATLAB, and can be used in a Pythonic, object oriented way. Although Matplotlib is written primarily in pure Python, it makes heavy use of NumPy and other extension code to provide good performance even for large arrays. Matplotlib is designed with the philosophy that you should be able to create simple plots with just a few commands, or just one! If you want to see a histogram of your data, you shouldn't need to instantiate objects, call methods, set properties, and so on; it should just work."

62. For more on `matplotlib`'s development and history: John Hunter, "[History](https://matplotlib.org/users/history.html)" Matplotlib (2008)

### `SciPy`

63. "SciPy is a free and open-source Python library used for scientific computing and technical computing. SciPy contains modules for optimization, linear algebra, integration, interpolation, special functions, FFT, signal and image processing, ODE solvers and other tasks common in science and engineering. SciPy builds on the NumPy array object and is part of the NumPy stack which includes tools like Matplotlib, pandas and SymPy, and an expanding set of scientific computing libraries. This NumPy stack has similar users to other applications such as MATLAB, GNU Octave, and Scilab. The NumPy stack is also sometimes referred to as the SciPy stack" ([Wikipedia](https://en.wikipedia.org/wiki/SciPy)).

64. For more on `SciPy`, consult the [package documentation](https://www.scipy.org/scipylib/index.html).

## Machine Learning Libraries

65. So putting that all together, we have the following packages as the the foundation for our machine learning environment:
  * `NumPy`
  * `SciPy`
  * `matplotlib`
  * `pandas`

66. This combination of packages is part of what is known as the [`SciPy` stack](https://www.scipy.org/stackspec.html).

### Scikit-learn

67. "Scikit-learn...is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy" ([Wikipedia](https://en.wikipedia.org/wiki/Scikit-learn)).

68. For more on `scikit-learn`: https://scikit-learn.org

69. `scikit-learn` supports a wide range of supervised and unsupervised machine learning algorithms.

### Deep Learning

70. There are a few Python libraries specifically built for deep learning.

#### TensorFlow

71. Developed by the team at Google Brain, "TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications" (["Why TensorFlow"](https://www.tensorflow.org/).

#### Keras

72. "Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation" (["About Keras"](https://keras.io/about/)).

#### PyTorch

73. "PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab (FAIR). It is free and open-source software released under the Modified BSD license...A number of pieces of Deep Learning software are built on top of PyTorch, including Tesla Autopilot [and] Uber's Pyro" ([Wikipedia](https://en.wikipedia.org/wiki/PyTorch)).

74. For more on PyTorch: https://pytorch.org/

# Next Steps

75. [Click here](https://github.com/kwaldenphd/machine-learning-intro/blob/main/ml-python-lab.md) to go to the next section of this lab.
