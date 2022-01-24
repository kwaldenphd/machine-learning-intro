# Getting Started With Machine Learning in Python: Core Concepts in Artificial Intelligence

<a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license"><img style="border-width: 0;" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" alt="Creative Commons License" /></a>
This tutorial is licensed under a <a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license">Creative Commons Attribution-NonCommercial 4.0 International License</a>.


# Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Artificial Intelligence](#artificial-intelligence)
- [Machine Learning](#machine-learning)
  * [Predictive Modelling](#predictive-modelling)
    * [Predictive Analytics](#predictive-analytics)
  * [Machine Learning: Putting It All Together](#machine-learning-putting-it-all-together)
- [Deep Learning](#deep-learning)
  * [Neural Network](#neural-network)
  * [Deep Learning: Putting It All Together](#deep-learning-putting-it-all-together)
- [Machine Learning Versus Deep Learning](#machine-learning-versus-deep-learning)

# Overview

1. When working in and around artificial intelligence and machine learning, the jargon and concepts can quickly become overwhelming.

2. This tutorial is designed to provide an overview of core concepts in artificial intelligence, with a focus on machine learning.

# Algorithms

3. “In mathematics and computer science, an algorithm is a finite sequence of well-defined, computer-implementable instructions, typically to solve a class of problems or to perform a computation. Algorithms are always unambiguous and are used as specifications for performing calculations, data processing, automated reasoning, and other tasks.” ([Wikipedia](https://en.wikipedia.org/wiki/Algorithm))

<p align="center"><a href="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_1.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_1.png?raw=true" height="350" /></a></p>

<p align="center">Image from <a href="https://www.geeksforgeeks.org/introduction-to-algorithms/">Geeks for Geeks</a></p>

4. A recipe used in cooking is one example of an algorithm. The ingredients list and recipe steps provide a set of conditions and a step-by-step procedure for completing or achieving a specific desired outcome.

<p align="center"><a href="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_2.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_2.png?raw=true" /></a></p>
<p align="center">Image from <a href="https://www.webopedia.com/TERM/A/algorithm.html">Webopedia</a></p>

5. We can think of a flowchart as a visual representation of the steps or procedures of an algorithm. 

<p align="center"><a href="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_3.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_3.png?raw=true" /></a></p>
<p align="center">Image from <a href="https://www.c-programming-simple-steps.com/algorithm-definition.html">C-Programming</a></p>

<p align="center"><a href="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_4.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_4.png?raw=true" /></a></p>
<p align="center">TED-Ed, "<a href="https://youtu.be/6hfOvs8pY1k">What's an algorithm?</a>" <em>YouTube Video</em> (20 May 2013).</p>

6. We can think of the information contained in an algorithm as a type of pseudo-code that provides a set of instructions to the underlying machine. The compiler and assembler functions for the programming language we are using will “translate” the more human-readable algorithm into a discrete set of instructions for the machine.

# Artificial Intelligence

7. “In computer science, artificial intelligence (AI), sometimes called machine intelligence, is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of ‘intelligent agents’: any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term ‘artificial intelligence" is often used to describe machines (or computers) that mimic ‘cognitive’ functions that humans associate with the human mind, such as ‘learning’ and ‘problem solving.’” ([Wikipedia](https://en.wikipedia.org/wiki/Artificial_intelligence))

# Machine Learning

8. “Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as ‘training data,’ in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop a conventional algorithm for effectively performing the task.” ([Wikipedia](https://en.wikipedia.org/wiki/Machine_learning))

9. Machine learning relies on a branch of statistics to detect patterns or relationships within an input dataset. 

10. Those statistically significant patterns/relationships are used to classify or categorize future events or data points, based on a statistical measure of propability.

11. A machine learning algorithm is the procedure of set of instructions used by the model to make decisions about patterns or relationships.

## Predictive Modelling

12. "Predictive modelling uses statistics to predict outcomes. Most often the event one wants to predict is in the future, but predictive modelling can be applied to any type of unknown event, regardless of when it occurred...In many cases the model is chosen on the basis of detection theory to try to guess the probability of an outcome given a set amount of input data...Models can use one or more classifiers in trying to determine the probability of a set of data belonging to another set...Depending on definitional boundaries, predictive modelling is synonymous with, or largely overlapping with, the field of machine learning, as it is more commonly referred to in academic or research and development contexts. When deployed commercially, predictive modelling is often referred to as predictive analytics. Predictive modelling is often contrasted with causal modelling/analysis. In the former, one may be entirely satisfied to make use of indicators of, or proxies for, the outcome of interest. In the latter, one seeks to determine true cause-and-effect relationships." ([Wikipedia](https://en.wikipedia.org/wiki/Predictive_modelling))

13. Let's break down that Wikipedia definition.

14. "Predictive modelling uses statistics to predict outcomes. Most often the event one wants to predict is in the future, but predictive modelling can be applied to any type of unknown event, regardless of when it occurred."

  * In machine learning, a predictive model is using a measure of statistical significance derived from sample or input data to make a prediction about a future data point.

15. "In many cases the model is chosen on the basis of detection theory to try to guess the probability of an outcome given a set amount of input data...Models can use one or more classifiers in trying to determine the probability of a set of data belonging to another set."

  * A machine learning algorithm develops a model that relies on measures of statistical significance in a given input dataset to determine probability or likelihood of a particular outcome not accounted for in the input dataset.

  * For example, let's say your input data is the types of fruit present in a fruit basket. 

  * A machine learning algorithm would take information about the fruit to develop a model that predicts the type of fruit for new items added to the basket.

16. "Depending on definitional boundaries, predictive modelling is synonymous with, or largely overlapping with, the field of machine learning, as it is more commonly referred to in academic or research and development contexts. When deployed commercially, predictive modelling is often referred to as predictive analytics"

  * More jargon!

  * While there are important distinctions between these terms (machine learning, predictive modelling, and predictive analytics), the Venn diagram gets messy quickly.

### Predictive Analytics

17. “Predictive analytics encompasses a variety of statistical techniques from data mining, predictive modelling, and machine learning, that analyze current and historical facts to make predictions about future or otherwise unknown events. In business, predictive models exploit patterns found in historical and transactional data to identify risks and opportunities. Models capture relationships among many factors to allow assessment of risk or potential associated with a particular set of conditions, guiding decision-making for candidate transactions.” ([Wikipedia](https://en.wikipedia.org/wiki/Predictive_analytics))

18. Back to predictive modelling.

19. "Predictive modelling is often contrasted with causal modelling/analysis. In the former, one may be entirely satisfied to make use of indicators of, or proxies for, the outcome of interest. In the latter, one seeks to determine true cause-and-effect relationships."

20. The last part of this definition is important when we think about the wide range of machine learning applications.

21. The predictive modelling used in machine learning is not designed or intended to determine/detect any type of causal relationship.

22. The machine learning algorithm takes the input dataset values as a proxy or indicator for the underlying phenomenon being observed/analyzed/predicted.

23. A machine learning model is not making any claims about causal relationships between data points and model output.

24. This contrasts with branches of statistics that seek to build models that are determining causal or cause-and-effect relationships.

## Machine Learning: Putting It All Together

<p align="center"><a href="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_5.jpg?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_5.jpg?raw=true" /></a></p>
<p align="center">Image from “<a href="https://medium.com/@chethankumargn/artificial-intelligence-definition-types-examples-technologies-962ea75c7b9b">Artificial Intelligence: Definition Types, Examples, Technologies</a>” Medium blog post</p>

25. Let’s look at how these different concepts and functions work together.

26. Machine learning uses predictive modeling to classify or categorize an unmarked dataset. 

<p align="center"><a href="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_6.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_6.png?raw=true" /></a></p>
<p align="center">Image from “<a href="https://medium.com/datadriveninvestor/exploring-machine-learning-f1dc6f3ec902">Exploring Machine Learning</a>” Medium blog post</p>

27. In this example, the raw input data includes three different types of fruit. The machine learning model interprets, processes, and classifies (or categorizes) the data to create the trained model in the algorithm’s output.

28. Other types of machine learning algorithms take a set of training data and use predictive modeling to classify or categorize the data.

<p align="center"><a href="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_7.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_7.png?raw=true" /></a></p>
<p align="center">Image from “<a href="https://medium.com/@canburaktumer/machine-learning-basics-with-examples-part-2-supervised-learning-e2b740ff014c">Machine Learning Basics</a>” Medium blog post</p>

29. In this example, a set of already classified training data is given to the machine learning algorithm. The accuracy of the algorithm is tested by giving new raw or unlabeled data to the algorithm to see if it can classify it correctly based on the training data.

# Deep Learning

30. In recent years, deep learning has emerged as an advanced application of machine learning in artificial intelligence.

31. “Deep learning (also known as deep structured learning or differential programming) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Deep learning is a class of machine learning algorithms that uses multiple layers to progressively extract higher level features from the raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters or faces.” ([Wikipedia](https://en.wikipedia.org/wiki/Deep_learning))

## Neural Network

32. "Artificial neural networks (ANNs), usually simply called neural networks (NNs), are computing systems vaguely inspired by the biological neural networks that constitute animal brains. An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron that receives a signal then processes it and can signal neurons connected to it. The 'signal' at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate signal crosses that threshold. Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer), to the last layer (the output layer), possibly after traversing the layers multiple times." ([Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network))

## Deep Learning: Putting It All Together

<p align="center"><a href="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_8.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_8.png?raw=true" height="500"/></a></p>

<p align="center">Model of the type of neural network used in deep learning. <a href="https://en.wikipedia.org/wiki/Neural_network#/media/File:Neural_network_example.svg">Image from Wikimedia Commons</a></p>

<p align="center"><a href="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_9.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_9.png?raw=true" /></a></p>

<p align="center">Another model that illustrates a basic neural network and the type of neural network used in deep learning. Image from Xing, Wanli & Du, Dongping. (2018). Dropout Prediction in MOOCs: Using Deep Learning for Personalized Intervention. Journal of Educational Computing Research. DOI: <a href="https://www.researchgate.net/publication/323784695_Dropout_Prediction_in_MOOCs_Using_Deep_Learning_for_Personalized_Intervention">10.1177/0735633118757015</a>).</p>

# Machine Learning Versus Deep Learning

<p align="center"><a href="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_10.png?raw=true"><img class="aligncenter" src="https://github.com/kwaldenphd/artificial-intelligence-overview/blob/main/figures/Figure_10.png?raw=true" /></a></p>

<p align="center">Figure that illustrates the difference between machine learning and deep learning. <a href="https://www.quora.com/What-is-the-difference-between-deep-learning-and-usual-machine-learning">Image from Quora</a></p>
