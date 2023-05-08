import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load train and test datasets
train_df = pd.read_csv('imdb_reviews_small_lemm_train.tsv', sep='\t')
test_df = pd.read_csv('imdb_reviews_small_lemm_test.tsv', sep='\t')

# Extract lemmatized reviews and target labels from the train dataset
X = train_df['review_lemm']
y = train_df['pos']

# Split the train dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract lemmatized reviews from the test dataset
X_test = test_df['review_lemm']

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform train, validation, and test datasets using the vectorizer
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train_tfidf, y_train)

# Make predictions on the validation dataset
y_val_pred = clf.predict(X_val_tfidf)

# Calculate the model's accuracy on the validation dataset
accuracy = accuracy_score(y_val, y_val_pred)
print("Accuracy on validation set:", accuracy)

# Make predictions on the test dataset
y_pred = clf.predict(X_test_tfidf)

# Add the predictions to the test dataset and save as a CSV file
test_df['pos'] = y_pred
test_df.to_csv('predictions', index=False)
