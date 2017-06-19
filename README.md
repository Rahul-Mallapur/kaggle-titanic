# Kaggle: Titanic Dataset
Predicted the survival on Titanic dataset using regression, SVMs and ensemble methods. 

Libraries: ScikitLearn, Pandas, Numpy

## Implementation details

#### (a) Feature engineering:

A lot of interesting features are hidden in the dataset. For example, using the Ticket feature, one can extract the cabin -  which can better predict the chance of survival. Similarly instead of using the feature values directly, bagging within a feature proved to be useful, eg. the FamilySize variable. 

Missing values were appropriately replaced by medians or averages of similar groups. For the missing Age data, I found similar passengers using their Title, Gender, Class and averaged their values. 

Important features (out of the 67) were selected using the sklearn feature selection module.


#### (b) Model and predict:

My aim of the exercise was to research and learn the common classification models. Hence I liberally implemented different models and tweaked the hyperparameters. Models used: LogisticRegression, SVMs, KNN, RandomForest, AdaBoost. 

I got the best results on RandomForest, so further tuned the hyperparameters using GridSearchCV.
