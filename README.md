# Recipe Similarity Study Analysis

## Overview

This repository contains the analysis code and data for a study on recipe similarity, aimed at understanding human-perceived similarity using various features. The study involves analyzing recipe data, participant survey data, and human similarity judgments collected through an odd-one-out task.

## Repository Structure

- **data/**: Contains all the data files used in the study.
  - **images/**: Includes images of the recipes.
  - **ch&ge_samples_fv.csv**: Low-level image features vectors.
  - **participant_data.json**: Participant survey data.
  - **recipes.json**: Recipe data including titles, ingredients, and directions.
  - **similarity_judgements.json**: Results from the odd-one-out task.

- **results/**: Stores the generated results.
  - **similarity_matrices/**: Contains the similarity matrices.

- **scripts/**: Contains all the analysis scripts, organized by functionality.
  - **human_judgements/**: Scripts to compute the human judgements similarity matrix and a combination count of recipes appeared in the odd-one-out task.
  - **participant_analysis/**: Script for analyzing participant data.
  - **similarity_functions/**: Scripts computing similarity functions for recipe features and the correlation with human similarity judgements. 

## Setup

1. Clone the repository:


2. Create a virtual environment and install the dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. Run the analysis scripts as needed:

    ```bash
    python scripts/human_judgements/chosen_script.py
    ```
