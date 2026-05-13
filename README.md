# Amazon Kindle Book Review Sentiment Analysis

Sentiment analysis is the task of automatically identifying the emotional tone of a piece of text—typically classified as positive, negative, or neutral—using Natural Language Processing (NLP) and machine learning techniques.

In this project, we analyze the dataset [Amazon Kindle Book Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis), which contains user reviews along with ratings from 1 to 5.

To formulate a clear binary classification problem:
- Reviews with rating 3 (neutral) are removed  
- Ratings 1–2 → Negative (0)  
- Ratings 4–5 → Positive (1)  

We then train a Logistic Regression model and optimize it using GridSearchCV for hyperparameter tuning. The final model achieves an accuracy of approximately 86%, demonstrating solid performance for this task.

This project covers the full pipeline of a typical NLP workflow, including:
- Data preprocessing  
- Feature extraction  
- Model training  
- Model evaluation  

---

## Objectives

- Understand the fundamentals of machine learning  
- Explore different model configurations and hyperparameters  
- Evaluate model performance using appropriate metrics  
- Build a complete NLP pipeline from raw data to predictions  

---

## Technologies Used

- Python  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## Methodology

### 01_data_clean.ipynb
- Data cleaning and preprocessing  
- Text normalization (lowercase, removal of punctuation)  
- Removal of neutral reviews (rating = 3)  
- Mapping ratings to binary labels:
  - Negative → 0  
  - Positive → 1  
- Creation of a labeled sentiment column  

### 02_data_analysis.ipynb
- Exploratory Data Analysis (EDA)  
- Visualizations include:
  - Number of reviews per year  
  - Overall sentiment distribution  
  - Sentiment distribution over time  

### 03_data_training.ipynb
- Model training using Logistic Regression  
- Pipeline creation including:
  - Text vectorization (TF-IDF)  
  - Stopword removal (English)  
- Hyperparameter tuning using GridSearchCV  
- Selection of best-performing model based on accuracy  

### 04_data_testing.ipynb
- Evaluation of the trained model on synthetic test data  
- 10 sample reviews generated using ChatGPT  
- Prediction of sentiment labels for each review  

---

## Results

### Model Performance

- Accuracy: 86%  

#### Precision
- Positive: 88%  
- Negative: 83%  

#### Recall
- Positive: 89%  
  - Correctly predicted 5340 out of 6000 positive samples  
- Negative: 82%  
  - Correctly predicted 3280 out of 4000 negative samples  

#### F1 Score
- Positive: 88%  
- Negative: 82%  

---

### Visualizations

The project includes the following visual analyses:
- Number of reviews per year
  <p align="center">
  <img src="/data/plots/reviews_per_year.png" width="80%">
  </p>  
- Overall sentiment distribution
  <p align="center">
  <img src="/data/plots/sentiment_distribution.png" width="80%">
  </p>  
- Sentiment distribution per year
  <p align="center">
  <img src="/data/plots/sentiment_distribution_per_year.png" width="80%">
  </p>    
- Confusion matrix
  <p align="center">
  <img src="/data/plots/confusion_matrix.png" width="80%">
  </p>    

---

### Sample Test Reviews

The following synthetic reviews were used to evaluate the model:

1. "I absolutely loved this book! The story was gripping from start to finish."  
2. "One of the worst books I've ever read. The plot makes no sense."  
3. "It was an okay read, nothing special but not terrible either."  
4. "Fantastic writing style and deeply engaging characters. Highly recommended!"  
5. "I struggled to finish it. Very boring and repetitive."  
6. "A masterpiece. The author did an amazing job building the world and characters."  
7. "Not my cup of tea. I found it dull and predictable."  
8. "Really enjoyable book with a strong ending. I would read it again."  
9. "Poorly written and hard to follow. I expected much more."  
10. "An average book with some good moments but also some weak parts."  

The results of the model were: 

Review 1: 'I absolutely loved this book! The story was gripping from start to finish.'
Predicted sentiment: Positive

Review 2: 'One of the worst books I've ever read. The plot makes no sense.'
Predicted sentiment: Negative

Review 3: 'It was an okay read, nothing special but not terrible either.'
Predicted sentiment: Negative

Review 4: 'Fantastic writing style and deeply engaging characters. Highly recommended!'
Predicted sentiment: Positive

Review 5: 'I struggled to finish it. Very boring and repetitive.'
Predicted sentiment: Negative

Review 6: 'A masterpiece. The author did an amazing job building the world and characters.'
Predicted sentiment: Positive

Review 7: 'Not my cup of tea. I found it dull and predictable.'
Predicted sentiment: Negative

Review 8: 'Really enjoyable book with a strong ending. I would read it again.'
Predicted sentiment: Positive

Review 9: 'Poorly written and hard to follow. I expected much more.'
Predicted sentiment: Negative

Review 10: 'An average book with some good moments but also some weak parts.'
Predicted sentiment: Positive

As one can see, the only review that it got wrong was review 10.

---

## Future Improvements

- Experiment with more advanced models (e.g., SVM, Random Forest, or Transformers)

## Challenges and Learning 

- Data cleaning with pandas
- Creating visualization with matplotlib and seaborn
- Training and testing the model with scikit-learn

## How to Run the Project

### 0. Kaggle API setup 
This project downloads datasets using the Kaggle API. Since Kaggle credentials are personal, you must provide your own API key.

#### Generate your API key 

 - Go to (https://www.kaggle.com/settings/api)[https://www.kaggle.com/settings/api]
 - Scroll to the API section
 - Click on "**Create Legacy API key**"
 - This will download a file name ```kaggle.json```.

#### Place de credentials file 

- Put you ```kaggle.json``` in the folder ```kaggle```

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks
Start Jupyter Notebook:
```bash
jupyter notebook
```
Then run the notebooks in the following order:

1. ```01_data_clean.ipynb```
2. ```02_data_analysis.ipynb```
3. ```03_data_training.ipynb```
4. ```04_data_testing.ipynb```
