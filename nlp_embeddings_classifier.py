import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer

# Load and prepare the dataset
data = pd.read_csv('/Users/rams/ml-toy/spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

# Load SentenceTransformer model (you can change the model)
model_name = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name)

# Encode sentences into dense vectors
X_train_embed = embedder.encode(X_train.tolist(), convert_to_numpy=True)
X_test_embed = embedder.encode(X_test.tolist(), convert_to_numpy=True)

# Train a classifier
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
clf.fit(X_train_embed, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_embed)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))