# Word2Vec Text Similarity Checker (How I Met Your Mother Dataset)

## Project Overview
This project is a **Word2Vec-based Text Similarity Checker** built using the **How I Met Your Mother (HIMYM)** show transcript dataset. The goal of this project is to train a **Word2Vec** model to capture word embeddings and explore semantic similarities between characters and dialogues in the show. We also visualize word embeddings using PCA to understand relationships in a 3D space.

## Project Structure
The project consists of the following key components:
1. **Data Preprocessing**: 
   - Tokenization of text.
   - Conversion to lowercase.
   - Stopword removal.
2. **Word2Vec Model Training**:
   - Training a Word2Vec model to learn word embeddings from the dataset.
3. **Similarity Check**:
   - Calculating semantic similarity between words (e.g., between characters' names like `barney` and `ted`).
4. **Visualization**:
   - Using PCA to reduce the word embedding dimensions and visualizing them in 3D.

---

## Dataset
The dataset used in this project is a collection of transcripts from the popular TV show *How I Met Your Mother*. Each line from the show is tokenized and preprocessed to create training data for the Word2Vec model.

---

## Technologies Used
- **Python**: General scripting language.
- **NLTK**: Natural Language Toolkit for text processing (tokenization, stopword removal).
- **Gensim**: Library for unsupervised learning of word embeddings using Word2Vec.
- **Scikit-learn**: Used for PCA (Principal Component Analysis) to reduce the dimensionality of word vectors.
- **Plotly**: Interactive plotting library for 3D visualization of word embeddings.

---

## Preprocessing Steps
The following preprocessing steps were applied to the text data:
1. **Tokenization**: Splitting sentences into individual words.
2. **Lowercasing**: Converting all text to lowercase to ensure uniformity.
3. **Stopword Removal**: Removing common English stopwords to focus on meaningful words.

### Sample Preprocessing Code:
```python
def lowercase_text(texts):
    return [[word.lower() for word in text] for text in texts]

def remove_stopwords(texts):
    stop_words = set(stopwords.words("english"))
    return [[word for word in text if word not in stop_words] for text in texts]
