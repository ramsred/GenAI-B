import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Sample sentences
sentences = [
    "I love machine learning.",
    "Deep learning is a branch of AI.",
    "Natural Language Processing is fun.",
    "Python is great for data science.",
    "This is a spam message!",
    "Free money offer, click now!",
    "You have won a prize, claim now.",
    "Important update regarding your account.",
    "Let's schedule a meeting for tomorrow.",
    "Can we catch up later this week?"
]

# Step 2: Load pretrained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Convert sentences to embeddings
embeddings = model.encode(sentences)

# Step 4: Reduce dimensions using PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)
# [val1,val2] - for one sentence

# Step 5: Plot the reduced embeddings
plt.figure(figsize=(10, 6))
for i, sentence in enumerate(sentences):
    x, y = reduced[i]
    plt.scatter(x, y, marker='o')
    plt.text(x + 0.01, y + 0.01, sentence, fontsize=9)

plt.title("Sentence Embeddings Visualized using PCA")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()