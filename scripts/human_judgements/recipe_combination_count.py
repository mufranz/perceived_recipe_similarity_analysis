import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
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

# Initialize combination count matrix
num_recipes = len(consistent_recipe_order)
combination_counts = np.zeros((num_recipes, num_recipes))

# Process odd-one-out responses
for entry in data:
    available_images = [img.split('/')[-1] for img in entry["availableImages"]]
    indices = [recipe_to_index[img] for img in available_images]
    
    # Increment counts for each pair recipes
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            combination_counts[indices[i], indices[j]] += 1
            combination_counts[indices[j], indices[i]] += 1

recipe_titles = [image_to_title.get(img, img) for img in consistent_recipe_order]


def wrap_and_handle_special_characters(title, width=20):
    title = title.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    return '\n'.join(textwrap.wrap(title, width=width))

wrapped_titles = [wrap_and_handle_special_characters(title) for title in recipe_titles]

combination_df = pd.DataFrame(combination_counts, index=wrapped_titles, columns=wrapped_titles)

print(combination_df)

# Create and save count matrix
output_dir = '../../results/similarity_matrices'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(15, 12))
heatmap = sns.heatmap(combination_df, annot=True, fmt=".0f", cmap="viridis", cbar_kws={'label': 'Count'})
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=8)
plt.title('Recipe Appearance Combination Count')
plt.savefig(os.path.join(output_dir, 'recipe_appearance_combination_count.png'))


# Print additional information
flattened_counts = combination_counts[np.triu_indices(num_recipes, k=1)]
min_count = np.min(flattened_counts)
max_count = np.max(flattened_counts)
median_count = np.median(flattened_counts)
average_count = np.mean(flattened_counts)

print(f"Minimum appearance of recipe combinations: {min_count}")
print(f"Maximum appearance of recipe combinations: {max_count}")
print(f"Median appearance of recipe combinations: {median_count}")
print(f"Average appearance of recipe combinations: {average_count}")
