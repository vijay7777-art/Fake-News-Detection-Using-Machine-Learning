# Fake News Detection using Machine Learning

## Project Overview
In this project, I built a Machine Learning system to classify news articles as **Fake** or **Real**.  
The goal of this project is to apply Natural Language Processing (NLP) techniques and compare multiple classification models.

---

## Dataset
This project uses the Fake and Real News Dataset from Kaggle:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset contains:
- Fake.csv (Fake news articles)
- True.csv (Real news articles)

The data was combined, labeled, shuffled, and preprocessed before training.

---

##  Technologies & Tools Used
- Python
- Pandas & NumPy (Data Handling)
- Scikit-learn (Machine Learning)
- TF-IDF Vectorizer (Text Feature Extraction)
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

---

##  Project Workflow
1. Loaded Fake and Real datasets
2. Added labels (0 = Fake, 1 = Real)
3. Combined and shuffled data
4. Text preprocessing and cleaning
5. Converted text to numerical features using TF-IDF
6. Split data into training and testing sets
7. Trained multiple classification models
8. Compared model performance
9. Implemented manual news prediction testing

---

## Model Performance
- Logistic Regression: ~98%
- Decision Tree: ~99%
- Random Forest: ~98%
- Gradient Boosting: ~99%

The models performed well in distinguishing fake and real news articles.

---

##  How to Run
1. Clone the repository
2. Install required libraries:
   pip install pandas numpy scikit-learn
3. Download dataset from Kaggle link above
4. Place Fake.csv and True.csv in the same folder
5. Run the Jupyter Notebook

---

##  Future Improvements
- Hyperparameter tuning
- Add Confusion Matrix visualization
- Deploy using Streamlit or Flask
- Try Deep Learning models (LSTM, BERT)




