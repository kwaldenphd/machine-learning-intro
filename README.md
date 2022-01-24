# Getting Started With Machine Learning in Python

<a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license"><img style="border-width: 0;" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" alt="Creative Commons License" /></a>
This tutorial is licensed under a <a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

## Lab Goals

This lab provides an overview of core concepts in machine learning as well as an introduction to machine learning in the Python programming environment. By the end of this lab, students will be able to:
- Understand and describe core concepts in artificial intelligence and machine learning
- Describe and compare supervised and unsupervised machine learning algorithms
- Describe common types of supervised machine learning methods
- Understand the core Python libraries used in machine learning, with a focus on the `SciPy` stack
- Write a Python machine learning program implementing a k-nearest neighbor classification algorithm

# Table of Contents

- [Overview](#overview)
- [Summary](#summary)
- [Additional Resources](#additional-resources)
- [Lab Notebook Questions](#lab-notebook-questions)
  

# Overview

This lab has three sections-
- [an overview of artificial intelligence core concepts and terminology](https://github.com/kwaldenphd/machine-learning-intro/blob/main/ai-overview.md)
- [a more detailed overview of machine learning concepts and terminology, with a focus on the Python programming language](https://github.com/kwaldenphd/machine-learning-intro/blob/main/ml-python-intro.md)
- [a tutorial with step-by-step instructions for building a machine learning classification model](https://github.com/kwaldenphd/machine-learning-intro/blob/main/ml-python-lab.md)

## Summary

1. This lab covered a lot of ground. Let's recap.

2. We started by unpacking some of the key terms used in machine learning, including supervised and unsupervised machine learning algorithms.

3. We also talked about some of the common statistical methods used in those algorithms, including classification and regression.

4. We then walked through an overview of the core Python libraries and packages used in machine learning, with a focus on the `SciPy` stack.

5. The last section of the lab focused on building a model that predicts iris species based on sepal and petal measurements.

6. For our three-class classification problem, the iris species were classes, and the species for an individual flower was its label.

7. We stored the measurements and labels in NumPy arrays.

8. Our model used the k-nearest neighbors classification algorithm and split labeled data into a training set and a test set.

9. We passed those training data parameters to the model.

10. We then used the test set data to calculate the model's accuracy.

11. Putting that all together:

```Python
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
```

12. The core components of this example (`fit`, `predict`, and `score`) are common across other `scikit-learn` supervised algorithms.

# Additional Resources

13. Later in the semester, you will have the opportunity to build your own machine learning model. 

14. The steps outlined in this lab can get you started, but there's obviously additional research needed when making decisions about approaches, methods, and models.

15. Starting with a method or concept's Wikipedia page can give you the broad strokes as well as links or citatations for additional resources.

16. Andreas C. Müller and Sarah Guide's accessible *Introduction to Machine learning With Python: A Guide for Data Scientists* (O'Reilly, 2017) is where I would start with additional research. 
- [Publisher website](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/)
- [Link to access via Google Drive](https://drive.google.com/file/d/1VHBuayX6PoZZrFaps-HLs3exXoLPSlSM/view?usp=sharing) (ND users only)
- [Code repository](https://github.com/amueller/introduction_to_ml_with_python)

17. [Manning Publications](https://www.manning.com/) has a number of titles on machine learning and deep learning that are also valuable starting places.

# Lab Notebook Questions

Q1: Write a narrative that documents and describes your experience working through this lab. What challenges did you face, and how did you solve them? What did you learn about machine learning through this lab? How are you thinking about machine learning differently after this lab? 
- You are welcome (but not required) to include Python code as part of that narrative.
