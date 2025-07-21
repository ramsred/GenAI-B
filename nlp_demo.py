import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

# Load the dataset
data = pd.read_csv('/Users/rams/ml-toy/spam.csv', encoding='latin-1')

# Drop unnecessary columns
data = data[['v1', 'v2']]

print(data.head())
# Rename columns for clarity
data.columns = ['label', 'message']

# Preprocess the labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into features and labels
X = data['message']
y = data['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(X_train_vectorized.shape, X_test_vectorized.shape)


# Train a Neural Network classifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))