# Churn Prediction

### Introduction ###
Churn prediction is a common binary-classification task, which aims to identify customers who are more likely to cancel their subscription.
It is widely used in banking, insurance companies, platforms that work on a subscription model and telephone companies, in order to
identify and eventually target at-risk customers.

The current project is concerned with predicting the probability of churn on unseen data, representing customers on which information on churn behavious is not yet available. It does so by leveraging on a training dataset, which respresents past customers, through a set of features. The considered instances are represented through binary, continuous, ordinal and categorical features.


### Summary ###

After a first exploration step, in which we visualize data through piecharts, boxplots, barplots and histograms, the data is preprocessed to assure that outliers are removed and categorical features are represented in a dummy format. Furthermore, some continuous variables are transformed by means of a Box-Cox transformation, to make them more suitable for models such as LDA and QDA, which assume a normal distribution in the data. Since the response variable "Exited", which represents whether a customer is a churner (1), or isn't (0), has unbalanced classes, we prefer AUC and other evaluation measures based on the confusion matrix to accuracy, such as Recall, Precision and F-measure.

A series of classifiers is trained on the training data. In order to tune the parameters, cross-validation is used on all models, and the best parameters are searched through GridSearch.
We start with simple classifiers, namely k-Nearest Neighbours and Logistic Regression, to move onto Decision Trees, Linear Discriminant Analysis, Quadratic Discriminant Analysis and ensamble methods.
As expected, ensamble methods yield the best results in terms of AUC, as they combine a set of weak learners either by consensus or by learning. By comparing the performance on the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) of the various methods, Stacking is found to be the best-performing model.

Finally, we predict the probability of being churners of new customers (unseen data), through Stacking, which combines three base learners (random forest, decision tree and gradient boosting) and one meta-learner (logistic regression).


### Notes ###

It should be noted that, in real-life contexts, misclassifications may have different costs, depending on the positive or negative class. Therefore, different thresholds (e.g. more or less conservative) for the classifiers may be preferred, in order to minimize costs and maximize benefits. A cost-benefit matrix, in this case, may prove useful to actually tune the models.


---

### Requirements to run the project ###

A virtual environment with:
- Python 3 ipykernel

The following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pickle

---

### How to run the project ###

Download the zip folder containing the .ipynb files and the datasets train.csv and test.csv. Modify the paths to the directory in the .ipynb files to your actual directory. 

The files should be run in this order:
1. Exploratory analysis
2. Logistic Regression
3. k-Nearest Neighbours
4. Decision tree
5. LDA
6. QDA
7. Random Forest
8. Adaboost
9. XGBoost
10. Stacking
11. Evaluation
12. Predictions
