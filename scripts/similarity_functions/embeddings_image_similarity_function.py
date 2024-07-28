import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import textwrap
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

#directory containing the recipe images
image_directory = '../../data/images'

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

# Load pre-trained VGG-16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Extract embeddings for each image
embeddings = []
for img_name in consistent_recipe_order:
    img_path = os.path.join(image_directory, img_name)
    img_array = load_and_preprocess_image(img_path)
    embedding = model.predict(img_array)
    embeddings.append(embedding.flatten())

embeddings = np.array(embeddings)

# Compute cosine similarity
similarity_matrix_embeddings = cosine_similarity(embeddings)

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
embeddings_flat = similarity_matrix_embeddings.flatten()

# Compute Spearman correlation between human judgments and image embedding similarity matrix
spearman_embeddings_zscore, _ = spearmanr(embeddings_flat, zscore_flat)

print(f"Spearman correlation between image embeddings and z-score matrices: {spearman_embeddings_zscore}")

# Visualization of similarity matrices
recipe_titles = [image_to_title[img_name] for img_name in consistent_recipe_order]
wrapped_titles = ['\n'.join(textwrap.wrap(title, width=20)) for title in recipe_titles]

output_dir = '../../results/similarity_matrices'
os.makedirs(output_dir, exist_ok=True)

similarity_df_embeddings = pd.DataFrame(similarity_matrix_embeddings, index=wrapped_titles, columns=wrapped_titles)

# Create and save similarity matrix for image embeddings 
plt.figure(figsize=(15, 12))
sns.heatmap(similarity_df_embeddings, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Similarity'})
plt.title('Image Embeddings Similarity Matrix')
plt.savefig(os.path.join(output_dir,'image_embeddings_similarity_matrix.png'))
plt.clf()
