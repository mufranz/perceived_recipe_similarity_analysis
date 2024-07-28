import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import manhattan_distances
from scipy.stats import spearmanr
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

# Load CSV file with low-level image features
csv_file_path = '../../data/ch&ge_samples_fv.csv'
df = pd.read_csv(csv_file_path)

# match image feature to recipe IDs
df['id'] = df['path'].apply(lambda x: x.split('/')[-1])
filtered_df = df[df['id'].isin(consistent_recipe_order)]
filtered_df = filtered_df.set_index('id').loc[consistent_recipe_order].reset_index()

# Extract low-level features
features = ["brightness", "sharpness", "contrast", "colourfulness", "entropy"]

# Compute Manhattan distances and similarity matrices for each feature
similarity_matrices = {}

for feature in features:
    feature_matrix = filtered_df[[feature]].values
    manhattan_dist = manhattan_distances(feature_matrix)
    manhattan_similarity = 1 - (manhattan_dist / np.max(manhattan_dist))
    similarity_matrices[feature] = manhattan_similarity


# Create DataFrames for visualization
recipe_titles = [image_to_title[img_name] for img_name in consistent_recipe_order]
wrapped_titles = ['\n'.join(textwrap.wrap(title.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss'), width=20)) for title in recipe_titles]

output_dir = '../../results/similarity_matrices'
os.makedirs(output_dir, exist_ok=True)

for feature, similarity_matrix in similarity_matrices.items():
    similarity_df = pd.DataFrame(similarity_matrix, index=wrapped_titles, columns=wrapped_titles)

    # Display the similarity matrix in the terminal
    print(f"Manhattan Similarity Matrix based on {feature} feature:")
    print(similarity_df)

    # Create and save matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
    plt.title(f'Image {feature} Similarity Matrixfeature')
    plt.savefig(os.path.join(output_dir,f'image_{feature}_similarity_matrix.png'))
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
    
    # Decrease the counts for combinations with the odd-one-out
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

# Flatten the matrix
zscore_flat = similarity_matrix_zscore.flatten()

# Compute Spearman correlation between human judgments and low-level features
correlations = {}

for feature, similarity_matrix in similarity_matrices.items():
    feature_flat = similarity_matrix.flatten()
    
    spearman_zscore, _ = spearmanr(feature_flat, zscore_flat)
    
    correlations[feature] = {
        'spearman_zscore': spearman_zscore
    }

# Print the correlation results
for feature, corr in correlations.items():
    print(f"Correlations for {feature} feature:")
    print(f"  Spearman correlation with z-score: {corr['spearman_zscore']}")
    print()
