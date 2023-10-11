from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('/content/GoogleNews-vectors-negative300.bin.gz', binary=True,unicode_errors='ignore', limit=1000000) 
wv.save_word2vec_format('vectors.csv')


# Load and preprocess data
phrases = load_phrases_from_csv("phrases.csv")
preprocessed_phrases = preprocess_phrases(phrases)

# Train Word2Vec model
model = Word2Vec(preprocessed_phrases, vector_size=100, window=5, min_count=1, sg=0)

# Calculate phrase embeddings
phrase_embeddings = calculate_phrase_embeddings(model, preprocessed_phrases)

# Batch execution: Calculate cosine similarity
similarity_matrix = cosine_similarity(phrase_embeddings, phrase_embeddings)

# On-the-Fly Execution
def find_closest_match(user_input):
    user_input = preprocess_input(user_input)
    user_embedding = calculate_phrase_embedding(model, user_input)
    
    similarity_scores = cosine_similarity([user_embedding], phrase_embeddings)[0]
    
    # Find the closest match
    closest_index = similarity_scores.argmax()
    closest_phrase = phrases[closest_index]
    similarity_score = similarity_scores[closest_index]
    
    return closest_phrase, similarity_score
