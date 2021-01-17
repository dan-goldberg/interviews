# Answers
Dan Goldberg - 2021-01-19

- [Answers](#answers)
- [Question 1: Which shortstop converted the most outs above average?](#question-1-which-shortstop-converted-the-most-outs-above-average)
  - [1 Methodology](#1-methodology)
    - [1.1 Feature Design](#11-feature-design)
    - [1.2 Candidate Models](#12-candidate-models)
    - [1.3 Training Method](#13-training-method)
    - [1.4 Model Evaluation & Selection](#14-model-evaluation--selection)
  - [2. Code](#2-code)
    - [2.1 ModelExperiment (utils.ml_training)](#21-modelexperiment-utilsml_training)
    - [2.2 ModelPersistance (utils.ml_utils)](#22-modelpersistance-utilsml_utils)
    - [2.3 Diamond (utils.viz_utils)](#23-diamond-utilsviz_utils)
- [Question 2: In addition to what’s included in the provided dataset, what variables or types of information do you think would be helpful in answering this question more effectively?](#question-2-in-addition-to-whats-included-in-the-provided-dataset-what-variables-or-types-of-information-do-you-think-would-be-helpful-in-answering-this-question-more-effectively)
- [Question 3: Other than the final leaderboard, what is one interesting or surprising finding you made?](#question-3-other-than-the-final-leaderboard-what-is-one-interesting-or-surprising-finding-you-made)

# Question 1: Which shortstop converted the most outs above average?

The leader in OAA in this dataset was \[playername\] with \[outs\] converted above average.

## 1 Methodology

Want to know the likelihood of the average shortstop making an out on a given play (i.e. 70%), and observe if the shortstop being evaluated made the play, turning that probabity into 100%, adding the differnce for that particular observation (100% - 70% = 30%). Multivariate input, probability as output, and learning some function from the input to the output that minimizes the log-loss objective. 

Programmed in python leveraging scikit-learn, tensorflow.keras, and Stan (pyStan). 

### 1.1 Feature Design

Goal was to design features that would be describe the difficulty of the play regardless of where the shortstop was positioned on the field (show picture of starting position).

<img src="https://github.com/dan-goldberg/interviews/blob/master/tbj/tbj_202101/assets/BaseballDiamondScale.jpg?raw=true" width="400" alt="My Image">

### 1.2 Candidate Models

- Individual Model With Probability Output (i.e. Logistic Regression, Neural Network w/Sigmoid Activation, GAM w/Logit Link)
- Ensemble of non-probabilistic classification models for bootstrapped probability score (i.e. Gradient Boosted or Random Forest Decision Trees)

(Discuss linear vs non-linear )

### 1.3 Training Method

For non Bayesian Models (no priors):

- k-fold Cross Validation for evaluating the log-loss objective (inner loop)
- Bayesian Optimization for Hyperparameter Tuning (outer loop)

For Bayesian Models (w/ priors on parameters):

- Using Stan to define model and NUTS optimizer. 
- Carefully select priors by simulating in output space.

### 1.4 Model Evaluation & Selection

- Want out-of-sample Log-Loss, regardless of whether the model was Bayesian or non-Bayesian

## 2. Code

My programming efforts focused on creating functionality to make useful visualizations of the data, a generic pipeline for training models, and a way to save models for evaluation and model selection. I also wanted to showcase my skills in building modelling code for production

### 2.1 ModelExperiment (utils.ml_training)

Implements the training scheme for non-Bayesian models, including the inner loop k-fold cross validation, outer loop bayesian optimization, and model saving. 

### 2.2 ModelPersistance (utils.ml_utils)

Saves models trained in baysian optimization experiment so it's easy to load the exact model, along with meta-data of the model like parameter settings and the value of objective funciton at those settings. This makes it easy to check on the results of an experiment and load the best model to re-train and deploy. 

### 2.3 Diamond (utils.viz_utils)

Built as part of exporing the dataset, this class provides a convenient way to plot coordinates on a to-scale diagram of a generic baseball diamond, along with line segments for ball trajectory and player trajectory. 

# Question 2: In addition to what’s included in the provided dataset, what variables or types of information do you think would be helpful in answering this question more effectively? 

- Number of outs at the time of play - impossible to model doubleplays without this (need to know if doubleplay is even possible, or if there's two outs).
- Actual position of interception - would help break up components of range, field, and throw. Would also be an important step for modelling probability of doubleplay.
- The handedness of the batter and the speed of the batter - to estimate the batters time to first base.
- The speed of the baserunners - to estimate the time to their destination base.
- The handedness of the pitcher, maybe.
- Accurate spin readings, plus 3D spin, not just 2D spin. 
- The ballpark - they might have different hop profiles for groundballs due to different materials (i.e. turf), groundskeeping, design of infield.

# Question 3: Other than the final leaderboard, what is one interesting or surprising finding you made? 

While exploring the shortstop defense dataset I became interested in the launch spin and launch axis measurements. These columns contain much missing data, and I thought that they might be useful for modelling the probability of the shortstop making an out - my hypothesis was that a groundball with top spin or side spin might behave much more eradically than a ball starting with backspin, and that more spin might make some plays like charging plays more difficult. I thought that I would try to impute the missing values in some way to try to utilize the useful information that the non-missing values might have for modelling out probability. To do this I created a model of spin using a few features I thought would be causally related to spin - launch speed and launch angle. 