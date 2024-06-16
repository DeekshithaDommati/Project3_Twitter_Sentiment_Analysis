import pandas as pd

# Load the dataset
file_path = 'C:/Users/domma/Downloads/archive/training.1600000.processed.noemoticon.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Display basic information about the dataset
print(data.info())
print(data.head())


# DATA EXPLORATION
# Rename columns
data.columns = ['sentiment', 'tweet_id', 'timestamp', 'query', 'user', 'tweet_content']

# Display basic information about the dataset
print(data.info())
print(data.head())


# DATA CLEANING
# Convert sentiment labels: 0 -> negative, 4 -> positive
data['sentiment'] = data['sentiment'].replace({0: 'negative', 4: 'positive'})

# Check for missing values
print(data.isnull().sum())

# Remove duplicate entries
data.drop_duplicates(inplace=True)

# Clean tweet content
import re

def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#', '', tweet)  # Remove hashtags
    tweet = re.sub(r'RT[\s]+', '', tweet)  # Remove RT
    tweet = re.sub(r'\W', ' ', tweet)  # Remove punctuation
    return tweet.lower()

data['cleaned_tweet'] = data['tweet_content'].apply(clean_tweet)


#EDA
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Plot sentiment distribution
sentiment_counts = data['sentiment'].value_counts()
plt.figure(figsize=(8,6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,hue=sentiment_counts.index, palette='viridis',legend=False)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Word cloud for all tweets
all_words = ' '.join([text for text in data['cleaned_tweet']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# SENTIMENT DISTRIBUTION
# already completed in eda part


# WORD FREQUENCY ANALYSIS
from collections import Counter

# Positive sentiment words
positive_words = ' '.join([text for text in data[data['sentiment'] == 'positive']['cleaned_tweet']])
positive_wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)

plt.figure(figsize=(10, 7))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Sentiment')
plt.show()

# Negative sentiment words
negative_words = ' '.join([text for text in data[data['sentiment'] == 'negative']['cleaned_tweet']])
negative_wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Negative Sentiment')
plt.show()


# TEMPORIAL ANALYSIS
# Remove timezone part from the timestamp and then convert to datetime
#data['timestamp'] = data['timestamp'].apply(lambda x: ' '.join(x.split()[:-1]))
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Resample to get sentiment counts per month
monthly_sentiment = data.resample('ME')['sentiment'].value_counts().unstack().fillna(0)

monthly_sentiment.plot(kind='line', figsize=(15, 7))
plt.title('Monthly Sentiment Trends')
plt.xlabel('Month')
plt.ylabel('Number of Tweets')
plt.show()


# TEXT PROCESSING
import nltk

# Download the stopwords corpus
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_tweet(tweet):
    tokens = word_tokenize(tweet)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

data['preprocessed_tweet'] = data['cleaned_tweet'].apply(preprocess_tweet)


#SENTIMENT PREDICTION MODEL
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Split data into training and testing sets
X = data['preprocessed_tweet']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vect)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred, average='weighted'))



# FEAUTURE IMPORTANCE
import numpy as np

feature_names = vectorizer.get_feature_names_out()
feature_importance = np.abs(model.coef_[0])
sorted_indices = np.argsort(feature_importance)[::-1]

top_features = 20
top_feature_names = [feature_names[i] for i in sorted_indices[:top_features]]
top_feature_importance = feature_importance[sorted_indices[:top_features]]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_feature_importance, y=top_feature_names, palette='viridis')
plt.title('Top Features Contributing to Sentiment Prediction')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Names')
plt.show()




