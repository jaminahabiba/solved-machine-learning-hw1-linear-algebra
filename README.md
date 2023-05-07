Download Link: https://assignmentchef.com/product/solved-machine-learning-hw1-linear-algebra
<br>
<h1>Linear Algebra</h1>

<ol>

 <li>A matrix <em>A </em>over R is called <em>positive semidefinite </em>(PSD) if for every vector <em>v</em>, <em>v</em><sup>T</sup><em>Av </em>≥ 0. Show that a symmetric matrix <em>A </em>is PSD if and only if it can be written as <em>A </em>= <em>XX<sup>T </sup></em>.</li>

</ol>

Hint: a real symmetric matrix <em>A </em>can be decomposed as <em>A </em>= <em>Q<sup>T </sup>DQ</em>, where Q is an orthogonal matrix and <em>D </em>is a diagonal matrix with eigenvalues of <em>A </em>as its diagonal elements.

<ol start="2">

 <li>Show that the set of all symmetric positive semidefinite matrices is <em>convex</em>: Namely, that for any two symmetric positive semidefinite matrices <em>A,B </em>and a scalar 0 ≤ <em>θ </em>≤ 1, <em>θA</em>+(1−<em>θ</em>)<em>B </em>is also symmetric positive semidefinite.</li>

</ol>

<h1>Calculus and Probability</h1>

<ol>

 <li><em>Matrix calculus </em>is the extension of notions from calculus to matrices and vectors. We define the derivative of a scalar <em>y </em>with respect to a vector <strong>x </strong>as the column vector which obeys:</li>

</ol>

for <em>i </em>= 1<em>,…,n</em>. Let <em>A </em>be a <em>n </em>× <em>n </em>matrix. Prove that: <strong>x</strong>

<ol start="2">

 <li>Let <strong>p </strong>= (<em>p</em><sub>1</sub><em>,…,p<sub>n</sub></em>) be a discrete distribution, with = 1 and <em>p<sub>i </sub></em>≥ 0 for <em>i </em>= 1<em>,…,n</em>.</li>

</ol>

The <em>entropy</em>, which measures the uncertainty of a distribution, is defined by:

<em>n</em>

<em>H</em>(<strong>p</strong>) = −<sup>X</sup><em>p<sub>i </sub></em>log<em>p<sub>i</sub></em>

<em>i</em>=1

where we define 0log0 = 0. Use Lagrange multipliers to show that the uniform distribution has the largest entropy (Tip: Ignore the inequality constraints <em>p<sub>i </sub></em>≥ 0 and show that the solution satisfies them regardless).

<ol start="3">

 <li>Let <em>X</em><sub>0</sub><em>,…,X<sub>n</sub></em><sub>−1 </sub>be <em>n </em>positive (∀<em>i,X<sub>i </sub></em>∈ [0<em>,</em>∞)) i.i.d random variables with a <strong>continuous </strong>probability function <em>f<sub>X</sub></em>. We will prove the following:</li>

</ol>

We will start with a side lemma that will help us with the proof.

<ul>

 <li>Prove the following lemma: .</li>

</ul>

Where <em>F<sub>X</sub></em>(<em>a</em>) is the cumulative distribution function (CDF) of <em>X</em><sub>0 </sub>at point <em>a </em>and <em>f<sub>X</sub></em><sub>0</sub>(<em>a</em>) is the probability density function (PDF) of <em>X</em><sub>0 </sub>at <em>a</em>.

<ul>

 <li>Recall that the CDF is an integral over the probability density function (<em>f </em>is continuous) and use this fact to complete the above theorem.</li>

</ul>

1

2                                                                                                                                      <em>Handout Homework 1: March 8, 2020</em>

<h1>Decision Rules and Concentration Bounds</h1>

<ol>

 <li>Let <em>X </em>and <em>Y </em>be random variables where <em>Y </em>can take values in {1<em>,…,L</em>}. Let <em>`</em><sub>0−1 </sub>be the 0-1 loss function defined in class. Show that <em>h </em>= arg min E[<em>`</em><sub>0−1</sub>(<em>f</em>(<em>X</em>)<em>,Y </em>)] is given by <em>f</em>:X→R</li>

 <li>Let <strong>X </strong>= (<em>X</em><sub>1</sub><em>,…,X<sub>n</sub></em>)<sup>T </sup>be a vector of random variables. <strong>X </strong>is said to have a <strong>multivariate normal (or Gaussian) distribution </strong>with mean <em>µ </em>∈ R<em><sup>n </sup></em>and a <em>n </em>× <em>n </em>positive definite covariance matrix Σ, if its probability density function is given by</li>

</ol>

<em>f</em>(<strong>x</strong>;<em>µ,</em>

where E[<em>X<sub>i</sub></em>] = <em>µ<sub>i </sub></em>and <em>cov</em>(<em>X<sub>i</sub>,X<sub>j</sub></em>) = Σ<em><sub>ij </sub></em>for all <em>i,j </em>= 1<em>,…,n</em>. We write this as <strong>X </strong>∼ N(<em>µ,</em>Σ).

In this question, we generalize the decision rule we have seen in the recitation to more than one dimension. Assume that the data is h<strong>x</strong><em>,y</em>i pairs, where <strong>x </strong>∈ R<em><sup>d </sup></em>and <em>y </em>∈ {0<em>,</em>1}. Denote by <em>f</em><sub>0</sub>(<strong>x</strong>) and <em>f</em><sub>1</sub>(<strong>x</strong>) the probability density functions of <strong>x </strong>given each of the label values. It is known that <em>f</em><sub>0</sub><em>,f</em><sub>1 </sub>are multivariate Gaussian:

<em>f</em><sub>0</sub>(<strong>x</strong>) = <em>f</em>(<strong>x</strong>;<em>µ</em><sub>0</sub><em>,</em>Σ) <em>f</em><sub>1</sub>(<strong>x</strong>) = <em>f</em>(<strong>x</strong>;<em>µ</em><sub>1</sub><em>,</em>Σ)

Note that the covariance matrix, Σ, is the same for both distributions, but the mean vectors, <em>µ</em><sub>0</sub><em>,</em><em>µ</em><sub>1</sub>, are different. Finally, it is known that the probability to sample a positive sample (i.e. <em>y </em>= 1) is <em>p</em>.

<ul>

 <li>We are given a point <strong>x </strong>and we need to label it with either <em>y </em>= 0 or <em>y </em>= 1. Suppose our decision rule is to decide <em>y </em>= 1 if and only if P[<em>y </em>= 1|<strong>X</strong>] <em>&gt; </em>P[<em>y </em>= 0|<strong>X</strong>]. Find a simpler condition for <strong>X </strong>that is equivalent to this rule.</li>

 <li>The decision boundary for this problem is defined as the set of points for which P[<em>y </em>= 1|<strong>X</strong>] = P[<em>y </em>= 0|<strong>X</strong>]. What is the shape of the decision a general <em>d &gt; </em>1 (think of <em>d </em>= 1 and <em>d </em>= 2 for intuition)?</li>

</ul>

<ol start="3">

 <li>Let <em>X</em><sub>1</sub><em>,…,X<sub>n </sub></em>be i.i.d random variables that are uniformly distributed over the interval [−3<em>,</em>5].</li>

</ol>

Define <em>S </em>= <em>X</em><sub>1 </sub>+ <em>… </em>+ <em>X<sub>n</sub></em>. Use Hoeffding’s inequality to find <em>N </em>∈ N such that for all <em>n </em>≥ <em>N</em>

P[<em>S &gt; n</em><sup>2 </sup>+ 0<em>.</em>2<em>n</em>] <em>&lt; </em>0<em>.</em>1

<ol start="4">

 <li>Suppose we need to build a load balancing device to assign a set of <em>n </em>jobs to a set of <em>m </em> Suppose the <em>j</em>-th job takes <em>L<sub>j </sub></em>time, 0 ≤ <em>L<sub>j </sub></em>≤ 1 (say, in seconds). The goal is to assign the <em>n </em>jobs to the <em>m </em>servers so that the load is as balanced as possible (i.e., so that the busiest server finishes as quickly as possible). Suppose each server works sequentially through the jobs that are assigned to it and finishes in time equal to the sum of job lengths assigned to the server. Let be the total sum of job lengths (assume <em>L &gt;&gt; m</em>). With perfect load balancing, each server would take <em>L/m </em>time. There are some good algorithms for this scenario, but we are interested in analyzing the case of random assignment of jobs to servers.</li>

</ol>

<em>Handout Homework 1: March 8, 2020                                                                                                                                      </em>3

Suppose we assign a random server for each job, with replacement. Denote by <em>R<sub>i,j </sub></em>the load on server <em>i </em>from job <em>j </em>– that is, <em>L<sub>j </sub></em>if server <em>i </em>was assigned for job <em>j</em>, or 0 otherwise. Also, let be the total load on server <em>i</em>.

<ul>

 <li>What is E[<em>R<sub>i</sub></em>]?</li>

 <li>We want to bound the probability that the load on the <em>i</em>-th server is more than <em>δ </em>= 10% larger than the expected load. Use the multiplicative form of the Chernoff bound to bound</li>

</ul>

P[<em>R<sub>i </sub></em>≥ (1 + <em>δ</em>) · E[<em>R<sub>i</sub></em>]]

Note that this form doesn’t require the summed random variables to be identically distributed.

<ul>

 <li>Now, we want to bound the probability that <strong>any </strong>of the servers are overloaded by more than <em>δ </em>= 10% of the expected load. Give a bound for:</li>

</ul>

<em> or … or </em>

using the results from (a) and (b) and using the union bound (reminder: for events

<em>A</em><sub>1</sub><em>,…,A<sub>k</sub></em>, the union bound is

4                                                                                                                                      <em>Handout Homework 1: March 8, 2020</em>

<h1>Programming Assignment</h1>

<ol>

 <li><strong>Nearest Neighbor. </strong>In this question, we will study the performance of the Nearest Neighbor (NN) algorithm on the MNIST dataset. The MNIST dataset consists of images of handwritten digits, along with their labels. Each image has 28×28 pixels, where each pixel is in grayscale scale, and can get an integer value from 0 to 255. Each label is a digit between 0 and 9. The dataset has 70,000 images. Althought each image is square, we treat it as a vector of size 784.</li>

</ol>

The MNIST dataset can be loaded with sklearn as follows:

&gt;&gt;&gt; from sklearn.datasets import fetch_openml

&gt;&gt;&gt; mnist = fetch_openml(’mnist_784’)

&gt;&gt;&gt; data = mnist[’data’]

&gt;&gt;&gt; labels = mnist[’target’]

Loading the dataset might take a while when first run, but will be immediate later. See http://scikit-learn.org/stable/datasets/mldata.html for more details. Define the training and test set of images as follows:

&gt;&gt;&gt; import numpy.random

&gt;&gt;&gt; idx = numpy.random.RandomState(0).choice(70000, 11000)

&gt;&gt;&gt; train = data[idx[:10000], :].astype(int)

&gt;&gt;&gt; train_labels = labels[idx[:10000]]

&gt;&gt;&gt; test = data[idx[10000:], :].astype(int)

&gt;&gt;&gt; test_labels = labels[idx[10000:]]

It is recommended to use numpy and scipy where possible for speed, especially in distance computations.

<ul>

 <li>Write a function that accepts as input: (i) a set of images; (ii) a vector of labels,corresponding to the images (ii) a query image; and (iii) a number <em>k</em>. The function will implement the <em>k</em>-NN algorithm to return a prediction of the query image, given the given label set of images. The function will use the <em>k </em>nearest neighbors, using the Euclidean L2 metric. In case of a tie between the <em>k </em>labels of neighbors, it will choose an arbitrary option.</li>

 <li>Run the algorithm using the first <em>n </em>= 1000 training images, on each of the test images, using <em>k </em>= 10. What is the accuracy of the prediction (measured by 0-1 loss; i.e. the percentage of correct classifications)? What would you expect from a completely random predictor?</li>

 <li>Plot the prediction accuracy as a function of <em>k</em>, for <em>k </em>= 1<em>,…,</em>100 and <em>n </em>= 1000. Discuss the results. What is the best <em>k</em>?</li>

 <li>Using <em>k </em>= 1, run the algorithm on an increasing number of training images. Plot the prediction accuracy as a function of <em>n </em>= 100<em>,</em>200<em>,…,</em> Discuss the results.</li>

</ul>