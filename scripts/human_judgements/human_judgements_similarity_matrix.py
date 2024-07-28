import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load recipes JSON file
with open('../../data/recipes.json', 'r', encoding='utf-8') as file:
    recipes_data = json.load(file)

# Create a mapping from image identifiers to titles
image_to_title = {recipe['image'].split('/')[-1]: recipe['title'] for recipe in recipes_data['recipes']}

# Load human judgements JSON file
with open('../../data/similarity_judgements.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

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

# Create mapping from recipe identifiers to indices
recipe_to_index = {recipe_id: idx for idx, recipe_id in enumerate(consistent_recipe_order)}

# Initialize similarity matrix
num_recipes = len(consistent_recipe_order)
similarity_counts = np.zeros((num_recipes, num_recipes))

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


# Apply Z-Score Normalization
mean = np.mean(similarity_counts)
std = np.std(similarity_counts)
if std > 0:
    similarity_matrix_zscore = (similarity_counts - mean) / std
else:
    similarity_matrix_zscore = similarity_counts

recipe_titles = [image_to_title.get(img, img) for img in consistent_recipe_order]

def wrap_and_handle_special_characters(title, width=20):
    title = title.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    return '\n'.join(textwrap.wrap(title, width=width))

wrapped_titles = [wrap_and_handle_special_characters(title) for title in recipe_titles]

similarity_df_zscore = pd.DataFrame(similarity_matrix_zscore, index=wrapped_titles, columns=wrapped_titles)

print("\nZ-Score Normalized Similarity Matrix:")
print(similarity_df_zscore)

# Create and save matrix
output_dir = '../../results/similarity_matrices'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(15, 12))
sns.heatmap(pd.DataFrame(similarity_df_zscore,  index=wrapped_titles, columns=wrapped_titles), annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Human Judgements Z-Score Normalized Similarity Matrix')
plt.savefig(os.path.join(output_dir, 'human_judgements_similarity_matrix.png'))
plt.clf()
