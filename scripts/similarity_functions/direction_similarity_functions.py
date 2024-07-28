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

# Create mapping from image identifiers to titles
image_to_title = {recipe['image'].split('/')[-1]: recipe['title'] for recipe in recipes_data['recipes']}

# Extract directions
directions = [recipe['directions'] for recipe in recipes_data['recipes'] if recipe['image'].split('/')[-1] in consistent_recipe_order]
directions = [' '.join(direction) for direction in directions]  # Join the list of directions into a single string

# TF-IDF Encoding
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(directions)
similarity_matrix_tfidf = cosine_similarity(tfidf_matrix)

# Convert to DataFram
recipe_titles = [image_to_title[img_name] for img_name in consistent_recipe_order]
wrapped_titles = ['\n'.join(textwrap.wrap(title.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss'), width=20)) for title in recipe_titles]
tfidf_similarity_df = pd.DataFrame(similarity_matrix_tfidf, index=wrapped_titles, columns=wrapped_titles)

print(tfidf_similarity_df)


# Topic Modeling (LDA)
num_topics = 100  # Set the number of topics to 100 as per the paper
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(directions)
lda_model = LDA(n_components=num_topics, max_iter=10, random_state=42)
lda_matrix = lda_model.fit_transform(count_matrix)
similarity_matrix_lda = cosine_similarity(lda_matrix)

# Convert to DataFrame 
lda_similarity_df = pd.DataFrame(similarity_matrix_lda, index=wrapped_titles, columns=wrapped_titles)

print(lda_similarity_df)


# Load human judgements JSON file
with open('../../data/similarity_judgements.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize similarity matrix
num_recipes = len(consistent_recipe_order)
similarity_counts = np.zeros((num_recipes, num_recipes))

# Map recipe image identifiers to indices
recipe_to_index = {recipe_id: idx for idx, recipe_id in enumerate(consistent_recipe_order)}

# Process the odd-one-out responses
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
tfidf_flat = similarity_matrix_tfidf.flatten()
lda_flat = similarity_matrix_lda.flatten()


# Compute Spearman correlations between human judgements and direction features
spearman_tfidf_zscore, _ = spearmanr(tfidf_flat, zscore_flat)

spearman_lda_zscore, _ = spearmanr(lda_flat, zscore_flat)

print(f"Spearman correlation between TF-IDF and z-score matrices: {spearman_tfidf_zscore}")

print(f"Spearman correlation between LDA and z-score matrices: {spearman_lda_zscore}")

# Visualization of similarity matrices
def wrap_and_handle_special_characters(title, width=20):
    title = title.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    return '\n'.join(textwrap.wrap(title, width=width))

wrapped_titles = [wrap_and_handle_special_characters(title) for title in recipe_titles]

# Create and matrices
output_dir = '../../results/similarity_matrices'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(15, 12))
sns.heatmap(pd.DataFrame(similarity_matrix_tfidf, index=wrapped_titles, columns=wrapped_titles), annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Directions TF-IDF Similarity Matrix')
plt.savefig(os.path.join(output_dir, 'directions_tfidf_similarity_matrix.png'))
plt.clf()

sns.heatmap(pd.DataFrame(similarity_matrix_lda, index=wrapped_titles, columns=wrapped_titles), annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Directions LDA Similarity Matrix')
plt.savefig(os.path.join(output_dir, 'directions_lda_similarity_matrix.png'))
plt.clf()

