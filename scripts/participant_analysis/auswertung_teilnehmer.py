import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load participant data
with open('../../data/participant_data.json', 'r') as file:
    data = json.load(file)
df = pd.json_normalize(data)

# Process age data
df['age'] = df['age'].astype(int)
bins = [18, 24, 34, 44, 54, float('inf')]
labels = ['18-24', '25-34', '35-44', '45-54', '>55']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

mean_age = df['age'].mean()
median_age = df['age'].median()

print(f"Durchschnittsalter: {mean_age:.2f}")
print(f"Medianalter: {median_age}")

# process gender data
df['gender'] = df['gender'].str.lower()
gender_order = ['male', 'female', 'other']
df['gender'] = pd.Categorical(df['gender'], categories=gender_order)

# Create and save charts
output_dir = '../../results/similarity_matrices'
os.makedirs(output_dir, exist_ok=True)

# Gender
plt.figure(figsize=(7, 5))
gender_counts = df['gender'].value_counts().reindex(gender_order)
plt.bar(gender_counts.index, gender_counts.values, color='black')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'participant_gender_distribution.png'))

# Age
plt.figure(figsize=(7, 5))
age_group_counts = df['age_group'].value_counts().sort_index()
plt.bar(age_group_counts.index, age_group_counts.values, color='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'participant_age_distribution.png'))


# Food Website Visits
plt.figure(figsize=(7, 5))
website_visits_order = ['never', '1-2-times-a-month', 'weekly', 'multiple-times-a-week', 'daily']
df['websiteVisits'] = pd.Categorical(df['websiteVisits'], categories=website_visits_order)
website_visits_counts = df['websiteVisits'].value_counts().reindex(website_visits_order)
plt.bar(website_visits_counts.index, website_visits_counts.values, color='black')
plt.xlabel('Food Website Visits')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'participant_website_visits.png'))

# Cooking Experience
plt.figure(figsize=(7, 5))
cooking_experience_order = ['1', '2', '3', '4', '5']
df['cookingExperience'] = pd.Categorical(df['cookingExperience'], categories=cooking_experience_order)
cooking_experience_counts = df['cookingExperience'].value_counts().reindex(cooking_experience_order)
plt.bar(cooking_experience_counts.index, cooking_experience_counts.values, color='black')
plt.xlabel('Cooking Experience')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'participant_cooking_experience_bw.png'))

# Eating Preference
plt.figure(figsize=(7, 5))
eating_preference_order = ['vegetarian', 'vegan', 'pescatarian', 'carnivore', 'none']
df['eatingHabit'] = pd.Categorical(df['eatingHabit'], categories=eating_preference_order)
eating_preference_counts = df['eatingHabit'].value_counts().reindex(eating_preference_order)
plt.bar(eating_preference_counts.index, eating_preference_counts.values, color='black')
plt.xlabel('Eating Preference')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'participant_eating_preference_bw.png'))

# Home Cooking Days 
plt.figure(figsize=(7, 5))
homecooked_meal_order = ['never', '1-2-times', '3-4-times', '5-6-times', 'daily']
df['homecookedMeal'] = pd.Categorical(df['homecookedMeal'], categories=homecooked_meal_order)
homecooked_meal_counts = df['homecookedMeal'].value_counts().reindex(homecooked_meal_order)
plt.bar(homecooked_meal_counts.index, homecooked_meal_counts.values, color='black')
plt.xlabel('Days Home Cooking (per week)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'participant_home_cooking_days_bw.png'))



# combined graphic
fig, axs = plt.subplots(3, 2, figsize=(18, 15))

# Gender
gender_counts = df['gender'].value_counts().reindex(gender_order)
axs[0, 0].bar(gender_counts.index, gender_counts.values, color='black')
axs[0, 0].set_xlabel('Gender')
axs[0, 0].set_ylabel('Count')
axs[0, 0].set_title('A', fontweight='bold')

# Age
age_group_counts = df['age_group'].value_counts().sort_index()
axs[0, 1].bar(age_group_counts.index, age_group_counts.values, color='black')
axs[0, 1].set_xlabel('Age')
axs[0, 1].set_ylabel('Count')
axs[0, 1].set_title('B', fontweight='bold')

# Food Website Visits 
website_visits_counts = df['websiteVisits'].value_counts().reindex(website_visits_order)
axs[1, 0].bar(website_visits_counts.index, website_visits_counts.values, color='black')
axs[1, 0].set_xlabel('Food Website Visits')
axs[1, 0].set_ylabel('Count')
axs[1, 0].set_title('C', fontweight='bold')
axs[1, 0].tick_params(axis='x', rotation=30)

# Days Home Cooking
homecooked_meal_counts = df['homecookedMeal'].value_counts().reindex(homecooked_meal_order)
axs[1, 1].bar(homecooked_meal_counts.index, homecooked_meal_counts.values, color='black')
axs[1, 1].set_xlabel('Days Home Cooking (per week)')
axs[1, 1].set_ylabel('Count')
axs[1, 1].set_title('D', fontweight='bold')

# Cooking Experience
cooking_experience_counts = df['cookingExperience'].value_counts().reindex(cooking_experience_order)
axs[2, 0].bar(cooking_experience_counts.index, cooking_experience_counts.values, color='black')
axs[2, 0].set_xlabel('Cooking Experience')
axs[2, 0].set_ylabel('Count')
axs[2, 0].set_title('E', fontweight='bold')

# Eating Preference 
eating_preference_counts = df['eatingHabit'].value_counts().reindex(eating_preference_order)
axs[2, 1].bar(eating_preference_counts.index, eating_preference_counts.values, color='black')
axs[2, 1].set_xlabel('Eating Preference')
axs[2, 1].set_ylabel('Count')
axs[2, 1].set_title('F', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'participant_combined_plots.png'))

