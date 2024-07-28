import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import Levenshtein
import difflib
import jellyfish
import nltk
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

nltk.download('punkt')

# Define recipe order
consistent_recipe_order = [
    "k_107204.jpg",  # Würziges Mangoldgemüse zu NT Putenbraten
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

# Load human judgements JSON file
with open('../../data/similarity_judgements.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


# Extract titles
titles = [image_to_title[img_name] for img_name in consistent_recipe_order]

# Initialize similarity matrixes
num_recipes = len(titles)
similarity_matrix_lv = np.zeros((num_recipes, num_recipes))
similarity_matrix_lcs = np.zeros((num_recipes, num_recipes))
similarity_matrix_jw = np.zeros((num_recipes, num_recipes))
similarity_matrix_bi = np.zeros((num_recipes, num_recipes))

# Calculate string-based similarities
for i in range(num_recipes):
    for j in range(num_recipes):
        if i != j:
            dist_lv = Levenshtein.distance(titles[i], titles[j])
            dist_lcs = difflib.SequenceMatcher(None, titles[i], titles[j]).ratio()
            dist_jw = jellyfish.jaro_winkler_similarity(titles[i], titles[j])
            bi_grams_i = list(ngrams(nltk.word_tokenize(titles[i]), 2))
            bi_grams_j = list(ngrams(nltk.word_tokenize(titles[j]), 2))
            common_bi_grams = set(bi_grams_i).intersection(set(bi_grams_j))
            if len(bi_grams_i) > 0 and len(bi_grams_j) > 0:
                dist_bi = len(common_bi_grams) / max(len(bi_grams_i), len(bi_grams_j))
            else:
                dist_bi = 0
            
            similarity_matrix_lv[i, j] = 1 - dist_lv / max(len(titles[i]), len(titles[j]))
            similarity_matrix_lcs[i, j] = dist_lcs
            similarity_matrix_jw[i, j] = dist_jw
            similarity_matrix_bi[i, j] = dist_bi
        else:
            similarity_matrix_lv[i, j] = 1
            similarity_matrix_lcs[i, j] = 1
            similarity_matrix_jw[i, j] = 1
            similarity_matrix_bi[i, j] = 1

#LDA topic modelling
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(titles)
lda_model = LDA(n_components=100, max_iter=10, random_state=42)
lda_matrix = lda_model.fit_transform(count_matrix)
similarity_matrix_lda = cosine_similarity(lda_matrix)


# Initialize similarity matrix
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
lv_flat = similarity_matrix_lv.flatten()
lcs_flat = similarity_matrix_lcs.flatten()
jw_flat = similarity_matrix_jw.flatten()
bi_flat = similarity_matrix_bi.flatten()
lda_flat = similarity_matrix_lda.flatten()

# Ensure lengths match
min_length = min(len(zscore_flat), len(lv_flat), len(lcs_flat), len(jw_flat), len(bi_flat), len(lda_flat))
zscore_flat = zscore_flat[:min_length]
lv_flat = lv_flat[:min_length]
lcs_flat = lcs_flat[:min_length]
jw_flat = jw_flat[:min_length]
bi_flat = bi_flat[:min_length]
lda_flat = lda_flat[:min_length]

#Spearman correlations
spearman_lv_zscore, _ = spearmanr(lv_flat, zscore_flat)
spearman_lcs_zscore, _ = spearmanr(lcs_flat, zscore_flat)
spearman_jw_zscore, _ = spearmanr(jw_flat, zscore_flat)
spearman_bi_zscore, _ = spearmanr(bi_flat, zscore_flat)
spearman_lda_zscore, _ = spearmanr(lda_flat, zscore_flat)

print(f"Spearman correlation between Levenshtein and z-score matrices: {spearman_lv_zscore}")
print(f"Spearman correlation between LCS and z-score matrices: {spearman_lcs_zscore}")
print(f"Spearman correlation between Jaro-Winkler and z-score matrices: {spearman_jw_zscore}")
print(f"Spearman correlation between Bi-Gram and z-score matrices: {spearman_bi_zscore}")
print(f"Spearman correlation between LDA and z-score matrices: {spearman_lda_zscore}")

# Visualization of similarity matrices
def wrap_and_handle_special_characters(title, width=20):
    title = title.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    return '\n'.join(textwrap.wrap(title, width=width))

wrapped_titles = [wrap_and_handle_special_characters(title) for title in titles]

# Create and save matrices
output_dir = '../../results/similarity_matrices'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(15, 12))
sns.heatmap(pd.DataFrame(similarity_matrix_lv, index=wrapped_titles, columns=wrapped_titles), annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Title Levenshtein Similarity')
plt.savefig(os.path.join(output_dir, 'title_levenshtein_similarity_matrix.png'))
plt.clf()

sns.heatmap(pd.DataFrame(similarity_matrix_lcs, index=wrapped_titles, columns=wrapped_titles), annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Title LCS Similarity')
plt.savefig(os.path.join(output_dir, 'title_lcs_similarity_matrix.png'))
plt.clf()

sns.heatmap(pd.DataFrame(similarity_matrix_jw, index=wrapped_titles, columns=wrapped_titles), annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Title Jaro-Winkler Similarity')
plt.savefig(os.path.join(output_dir, 'title_jaro_winkler_similarity_matrix.png'))
plt.clf()

sns.heatmap(pd.DataFrame(similarity_matrix_bi, index=wrapped_titles, columns=wrapped_titles), annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Title Bi-Gram Similarity')
plt.savefig(os.path.join(output_dir, 'title_bi_gram_similarity_matrix.png'))
plt.clf()

sns.heatmap(pd.DataFrame(similarity_matrix_lda, index=wrapped_titles, columns=wrapped_titles), annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Title LDA Similarity')
plt.savefig(os.path.join(output_dir, 'title_lda_similarity_matrix.png'))
plt.clf()

