# Amazon Kindle Book Review Sentiment Analysis

Sentiment analysis is the task of automatically identifying the emotional tone of a piece of text—typically classified as positive, negative, or neutral—using Natural Language Processing (NLP) and machine learning techniques.

In this project, we analyze the dataset [Amazon Kindle Book Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis), which contains user reviews along with ratings from 1 to 5. To create a clear binary classification problem, we remove all neutral reviews (rating 3) and map ratings 1–2 to **negative** and 4–5 to **positive**.

We then train a **Logistic Regression** model, optimizing its performance using **GridSearchCV** for hyperparameter tuning. The final model achieves an accuracy of approximately **86%**, demonstrating solid performance on this text classification task.

This project covers the full pipeline of a typical NLP workflow, including data preprocessing, feature extraction, model training, and evaluation.

## Objectives 

 - Learn the logic of machine learning, different machine learning algorithms, test of different parameters to obtain the better performance of the model and traing and testing of models.

## Methodology 

 - In the notebook ```01_data_clean.ipynb``` we clean the dataframes obtained in Kaggle. We normalize the reviews to lower case and without any ponctuation to assure that we have no problems. Additionaly, for our further analysis, we drop the rows with rating 3 and map the rows with rating 1 and 2 to 0 and rows with rating 4 and 5 to 1. Then we create a new column where the rating 0 are called negative and rating 1 are called positive.

 - In the notebook ```02_data_analysis.ipynb``` we perform a analysis for some informations contained in the dataframe. We plot graphs with informations such as number of reviews per year, absolute distribution of sentiments and distributions of sentiments per year.

 - In the notebook ```03_data_training.ipynb``` we train our model using the technique of Logistic Regression. We define a pipeline with steps such as vectorize to convert the reviews into numbers, defining the english stopwords. We use GridSearchCV to test the combinations of the parameters to obtain the combination with the best accuracy.

 - In the last notebook ```04_data_testing.ipynb``` we test the model by providing 10 reviews generated with the aid of ChatGPT to predict the sentiments of those reviews. 
