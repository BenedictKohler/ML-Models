# Extractive Summarization using TextRank Algorithm

# This is an example of creating bullet point summaries from multiple news articles

# Import needed libraries
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

dataframe = pd.read_csv("article_name.csv")

sentences = []
for s in dataframe['article_text']:
  sentences.append(sent_tokenize(s)) # Split into individual sentences

sentences = [y for x in sentences for y in x] # flatten list

# Each word in a sentence must be represented by a list/array of word embeddings
# Here we are using the Glove word embeddings but Bag of Words or others also work
# Word embeddings are a way of representing words by features, the central idea in the Glove model
# is that words that occur in a similar context and hence have similar meaning should have similar
# features. Thus, word embeddings are a way of giving similar words similar trainable feutures

# Here we are simply obtaining the embeddings/features for each word stored like below
# {'word': [feature1, feature2, feature3, feature4, ..., feature100]}
word_embeddings = {}
f = open('glove_word_embeddings.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

# It is generally good practice to clean the text first
# Remove everything except letters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# Convert to lowercase, so capitalized and uncapitalized words are treated the same
clean_sentences = [s.lower() for s in clean_sentences]

stop_words = stopwords.words('english')

# function to remove stopwords
# Stopwords are essentially words that add little meaning to the text such as "a", "and", "in", etc
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

# remove these stopwords
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


sentence_vectors = []
for sentence in clean_sentences:
  if len(sentence) != 0:
    # Each word is represented by a glove word embedding of 100 features
    v = sum([word_embeddings.get(word, np.zeros((100,))) for word in sentence.split()])/(len(sentence.split())+0.001) # 0.001 stops division by 0
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)
  
# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])

# We measure how similar sentences are to each other by using cosine similarity, which is the angle
# between the two vectos computed by arccos( (vec1 . vec2) / (||vec1|| * ||vec2||) )
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j: # Discard the same vectors
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

# We now convert similarity matrix into a graph, with nodes being the sentences and edges representing
# similarity scores between the sentences
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

# The edges represented the transition probabilities between sentences, meaning how similar they are to each other
# We reverse the result as the cosine_similarity is a measure between 0 and 1, with the more similar sentences
# being close to zero. The page rank however, ranks pages by highest (dissimilarity in this case), which is
# why we reverse it. To get most similar instead
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])
  