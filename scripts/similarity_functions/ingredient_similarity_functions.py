import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import textwrap
import re
from nltk.corpus import stopwords
import spacy

#Spacy German model
nlp = spacy.load("de_core_news_sm")

# Define recipe order
consistent_recipe_order = [
    "k_107204.jpg",  # Mangold Putenbraten
    "k_41465.jpg",   # Schweinerollbraten in Biersoße
    "k_112648.jpg",  # Putenleber Ludmilla
    "k_45291.jpg",   # Wirsing Kokosmilch (Hackfleisch)
    "k_16797.jpg",   # Filet Gemüseauflauf (Schweinefilet)
    "k_22791.jpg",   # Überbackene Knacker
    "k_36035.jpg",   # Spaghetti mit Lamm
    "k_24889.jpg",   # Kasslerpfanne (Fisch)
    "k_99462.jpg",   # Matjes (Fisch)
    "k_105923.jpg",  # Zucchini (Vegetarisch)
    "k_16743.jpg",   # Schupfnudel (Vegetarisch)
    "k_47728.jpg",   # Kartoffel Frischkäse Gratin (Vegetarisch)
    "k_68763.jpg",   # Champignons in Rahmsoße (Vegetarisch)
    "k_23157.jpg",   # Bulgur Salat (Vegan)
    "k_45101.jpg"    # Basilikum Knödel (Vegan)
]

# Load recipes JSON file
with open('../../data/recipes.json', 'r', encoding='utf-8') as file:
    recipes_data = json.load(file)

# Create a mapping from image identifiers to titles
image_to_title = {recipe['image'].split('/')[-1]: recipe['title'] for recipe in recipes_data['recipes']}
image_to_ingredients = {recipe['image'].split('/')[-1]: recipe['ingredients'] for recipe in recipes_data['recipes']}

# pre-processing ingredients 
# stop words and special characters
stop_words = set(stopwords.words('german'))
special_characters = re.compile('[^A-Za-z0-9äöüÄÖÜß ]+')
custom_adjectives = set(['frisch', 'geschnitten', 'gewürfelt', 'fein', 'entsteint', 'gemischt', 'mild'])

def preprocess_ingredient(ingredient):
    # Remove amounts and units
    ingredient = re.sub(r'\d+\s*[a-zA-Z]*', '', ingredient)
    # Remove special characters
    ingredient = special_characters.sub('', ingredient)
    # Process the ingredient with spacy
    doc = nlp(ingredient)
    # Remove stop words, custom adjectives, verbs, and adjectives
    filtered_tokens = [token.lemma_ for token in doc if token.text.lower() not in stop_words and token.text.lower() not in custom_adjectives and token.pos_ not in {'VERB', 'ADJ'}]
    # Join back into a single string
    return ' '.join(filtered_tokens)

# split ingredients
def split_conjunctions(ingredient):
    # Split compound ingredients
    if ' und ' in ingredient:
        return ingredient.split(' und ')
    elif ' oder ' in ingredient:
        return ingredient.split(' oder ')
    return [ingredient]

# Extract ingredients
ingredients_list = [image_to_ingredients[img_name] for img_name in consistent_recipe_order]
preprocessed_ingredients_list = []

for ingredients in ingredients_list:
    preprocessed_ingredients = []
    for ing in ingredients:
        processed_ingredient = preprocess_ingredient(ing['name'])
        split_ingredients = split_conjunctions(processed_ingredient)
        preprocessed_ingredients.extend(split_ingredients)
    preprocessed_ingredients_list.append(preprocessed_ingredients)


# binary ingredient vectors
ingredient_names = sorted(set(ing for ingredients in preprocessed_ingredients_list for ing in ingredients))
ingredient_index = {name: idx for idx, name in enumerate(ingredient_names)}

ingredient_vectors = np.zeros((len(consistent_recipe_order), len(ingredient_names)))

for i, ingredients in enumerate(preprocessed_ingredients_list):
    for ing in ingredients:
        ingredient_vectors[i, ingredient_index[ing]] = 1

# cosine similarity
similarity_matrix_cosine = cosine_similarity(ingredient_vectors)

# Jaccard similarity
ingredient_sets = [set(ingredients) for ingredients in preprocessed_ingredients_list]
similarity_matrix_jaccard = np.zeros((len(ingredient_sets), len(ingredient_sets)))

for i in range(len(ingredient_sets)):
    for j in range(len(ingredient_sets)):
        if i != j:
            intersection = len(ingredient_sets[i].intersection(ingredient_sets[j]))
            union = len(ingredient_sets[i].union(ingredient_sets[j]))
            similarity_matrix_jaccard[i, j] = intersection / union
        else:
            similarity_matrix_jaccard[i, j] = 1


#TF-IDF and LDA
ingredient_descriptions = [' '.join(ingredients) for ingredients in preprocessed_ingredients_list]

# TF-IDF Encodinh
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(ingredient_descriptions)
similarity_matrix_tfidf = cosine_similarity(tfidf_matrix)

# Topic Modeling (LDA)
num_topics = 100  
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(ingredient_descriptions)
lda_model = LDA(n_components=num_topics, max_iter=10, random_state=42)
lda_matrix = lda_model.fit_transform(count_matrix)
similarity_matrix_lda = cosine_similarity(lda_matrix)

# Convert to DataFrame
recipe_titles = [image_to_title[img_name] for img_name in consistent_recipe_order]
wrapped_titles = ['\n'.join(textwrap.wrap(title, width=20)) for title in recipe_titles]

cosine_similarity_df = pd.DataFrame(similarity_matrix_cosine, index=wrapped_titles, columns=wrapped_titles)
jaccard_similarity_df = pd.DataFrame(similarity_matrix_jaccard, index=wrapped_titles, columns=wrapped_titles)
tfidf_similarity_df = pd.DataFrame(similarity_matrix_tfidf, index=wrapped_titles, columns=wrapped_titles)
lda_similarity_df = pd.DataFrame(similarity_matrix_lda, index=wrapped_titles, columns=wrapped_titles)

# Print the similarity matrices
print("Cosine Similarity Matrix:")
print(cosine_similarity_df)
print("\nJaccard Similarity Matrix:")
print(jaccard_similarity_df)
print("\nTF-IDF Similarity Matrix:")
print(tfidf_similarity_df)
print("\nLDA Similarity Matrix:")
print(lda_similarity_df)

# Create and save matrices
output_dir = '../../results/similarity_matrices'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(15, 12))
sns.heatmap(cosine_similarity_df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Ingredient Cosine Similarity Matrix')
plt.savefig(os.path.join(output_dir,'ingredient_cosine_similarity_matrix.png'))
plt.clf()

sns.heatmap(jaccard_similarity_df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Ingredient Jaccard Similarity Matrix')
plt.savefig(os.path.join(output_dir,'ingredient_jaccard_similarity_matrix.png'))
plt.clf()

sns.heatmap(tfidf_similarity_df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Ingredient TF-IDF Similarity Matrix')
plt.savefig(os.path.join(output_dir,'ingredient_tfidf_similarity_matrix_ingredients.png'))
plt.clf()

sns.heatmap(lda_similarity_df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Ingredient LDA Similarity Matrix')
plt.savefig(os.path.join(output_dir,'ingredient_lda_similarity_matrix_ingredients.png'))
plt.clf()

# Load human judgements JSON file
with open('../../data/similarity_judgements.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize similarity matrix
num_recipes = len(consistent_recipe_order)
similarity_counts = np.zeros((num_recipes, num_recipes))

# Map recipe image identifiers to indices
recipe_to_index = {recipe_id: idx for idx, recipe_id in enumerate(consistent_recipe_order)}

# Process odd-one-out responses
for entry in data:
    chosen_image = entry["chosenImage"].split('/')[-1]
    available_images = [img.split('/')[-1] for img in entry["availableImages"]]
    
    chosen_idx = recipe_to_index[chosen_image]
    remaining_images = [img for img in available_images if img != chosen_image]
    idx1, idx2 = recipe_to_index[remaining_images[0]], recipe_to_index[remaining_images[1]]
    
    # Increase counts for similar pair
    similarity_counts[idx1, idx2] += 1
    similarity_counts[idx2, idx1] += 1
    
    # Decrease counts for combinations with the odd-one-out
    similarity_counts[chosen_idx, idx1] -= 1
    similarity_counts[idx1, chosen_idx] -= 1
    similarity_counts[chosen_idx, idx2] -= 1
    similarity_counts[idx2, chosen_idx] -= 1


# Apply Z-Score Standardization
mean = np.mean(similarity_counts)
std_dev = np.std(similarity_counts)
if std_dev > 0:
    similarity_matrix_zscore = (similarity_counts - mean) / std_dev
else:
    similarity_matrix_zscore = similarity_counts 

# Flatten the matrices
zscore_flat = similarity_matrix_zscore.flatten()
cosine_flat = similarity_matrix_cosine.flatten()
jaccard_flat = similarity_matrix_jaccard.flatten()
tfidf_flat = tfidf_matrix.toarray().flatten()
lda_flat = lda_matrix.flatten()

# lengths match
min_length = min(len(zscore_flat), len(cosine_flat), len(jaccard_flat), len(tfidf_flat), len(lda_flat))
zscore_flat = zscore_flat[:min_length]
cosine_flat = cosine_flat[:min_length]
jaccard_flat = jaccard_flat[:min_length]
tfidf_flat = tfidf_flat[:min_length]
lda_flat = lda_flat[:min_length]


# Spearman correlations between human judgements and ingredient functions
spearman_cosine_zscore, _ = spearmanr(cosine_flat, zscore_flat)
spearman_jaccard_zscore, _ = spearmanr(jaccard_flat, zscore_flat)
spearman_tfidf_zscore, _ = spearmanr(tfidf_flat, zscore_flat)
spearman_lda_zscore, _ = spearmanr(lda_flat, zscore_flat)

print(f"Spearman correlation between cosine and z-score matrices: {spearman_cosine_zscore}")
print(f"Spearman correlation between jaccard and z-score matrices: {spearman_jaccard_zscore}")
print(f"Spearman correlation between TF-IDF and z-score matrices: {spearman_tfidf_zscore}")
print(f"Spearman correlation between LDA and z-score matrices: {spearman_lda_zscore}")
