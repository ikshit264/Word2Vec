import gensim
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from nltk.corpus import stopwords
# import nltk

# Download necessary resources from NLTK
# nltk.download('punkt')
# nltk.download('stopwords')

# Data Loading
df = pd.read_csv(r"Day2\himym_full_transcript.csv")

lines = df['line']
story = [word_tokenize(line) for line in lines]

# Preprocessing functions
def lowercase_text(texts):
    # Convert all tokens to lowercase
    return [[word.lower() for word in text] for text in texts]

def remove_stopwords(texts):
    # Remove stopwords from tokenized text
    stop_words = set(stopwords.words("english"))
    return [[word for word in text if word not in stop_words] for text in texts]

def preprocessing(texts):
    # Apply preprocessing steps
    texts = lowercase_text(texts)
    texts = remove_stopwords(texts)
    return texts

# Preprocess the data
story = preprocessing(story)

# Initializing Word2Vec model
model = gensim.models.Word2Vec(
    vector_size=100,  # size of word vectors
    window=10,        # context window size
    min_count=2,      # ignore words that appear less than 2 times
    workers=4         # parallelization for faster training
)

# Build vocabulary
model.build_vocab(story)

# Train Word2Vec model
model.train(story, total_examples=model.corpus_count, epochs=model.epochs)

# Compute similarity between two words
print(f"Similarity between 'barney' and 'ted': {model.wv.similarity('barney', 'ted')}")
print(f"Similarity to 'barney': {model.wv.similar_by_word('barney')}")

# PCA for visualizing word embeddings in 3D
word_vectors = model.wv
words = model.wv.index_to_key  # List of words in the vocabulary
pca = PCA(n_components=3)
X = pca.fit_transform(word_vectors.vectors)

# Visualizing using Plotly 3D scatter plot
# fig = px.scatter_3d(X[200:300], x=0, y=1, z=2, color=words[200:300])
# fig.show()
