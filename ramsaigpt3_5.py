# -*- coding: utf-8 -*-
"""ramsAIgpt3.5.ipynb





# Import (bara 6%)
"""

import json
import pandas as pd
from collections import Counter

# List of JSON file paths
json_files = [
    '/content/recipes.json',
    '/content/baking.json',
    '/content/inspiration.json',
    '/content/health.json',
    '/content/budget.json',
]

# Initialize an empty list to store all recipes
all_recipes = []

# Load and append recipes from each file
for file_path in json_files:
    with open(file_path, 'r') as f:
        recipes = json.load(f)
        all_recipes.extend(recipes)  # Extend the list with recipes from this file

# Convert the combined list of recipes to a DataFrame
df = pd.DataFrame(all_recipes)

# Basic info
print(f"Total recipes: {len(df)}")
print("\nColumns and data types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

"""# Hreinsa/skoða gögn"""

#hreinsa gogn
def clean_recipe_data(df):
    # Fix typo in column name
    df.rename(columns={'rattings': 'ratings'}, inplace=True)

    # Handle missing nutrients - create empty dict if None
    df['nutrients'] = df['nutrients'].apply(lambda x: {} if pd.isna(x) else x)

    # Extract nutrient values into separate columns
    nutrient_cols = ['kcal', 'fat', 'saturates', 'carbs', 'sugars', 'fibre', 'protein', 'salt']
    for col in nutrient_cols:
        df[f'nutrient_{col}'] = df['nutrients'].apply(lambda x: x.get(col, None))

    # Convert numeric nutrient values to float (they're stored as strings)
    for col in nutrient_cols:
        df[f'nutrient_{col}'] = pd.to_numeric(df[f'nutrient_{col}'], errors='coerce')

    # Clean times data
    df['prep_time'] = df['times'].apply(lambda x: x.get('Preparation', None))
    df['cook_time'] = df['times'].apply(lambda x: x.get('Cooking', None))

    # Extract numeric values from time strings
    def extract_minutes(time_str):
        if pd.isna(time_str) or 'No Time' in str(time_str):
            return 0
        try:
            # Handle ranges like "15 mins - 20 mins"
            if '-' in time_str:
                return int(time_str.split('-')[0].strip().split(' ')[0])
            return int(time_str.split(' ')[0])
        except:
            return 0

    df['prep_time_mins'] = df['prep_time'].apply(extract_minutes)
    df['cook_time_mins'] = df['cook_time'].apply(extract_minutes)
    df['total_time_mins'] = df['prep_time_mins'] + df['cook_time_mins']

    # Clean serves data
    df['serves'] = pd.to_numeric(df['serves'], errors='coerce')

    return df

cleaned_df = clean_recipe_data(df.copy())

#analysis; skoða gogn
def analyze_recipe_data(df):
    # Basic stats
    print("\nRecipe difficulty distribution:")
    print(df['difficult'].value_counts())

    print("\nAverage ratings:", df['ratings'].mean())

    # Nutrient analysis
    print("\nAverage nutrition values per serving:")
    nutrient_cols = ['kcal', 'fat', 'saturates', 'carbs', 'sugars', 'fibre', 'protein', 'salt']
    for col in nutrient_cols:
        col_name = f'nutrient_{col}'
        print(f"{col}: {df[col_name].mean():.1f}")

    # Time analysis
    print("\nTime statistics (minutes):")
    print(f"Average prep time: {df['prep_time_mins'].mean():.1f}")
    print(f"Average cook time: {df['cook_time_mins'].mean():.1f}")
    print(f"Average total time: {df['total_time_mins'].mean():.1f}")

    # Category analysis
    print("\nMain categories:")
    print(df['maincategory'].value_counts())

    print("\nSubcategories:")
    print(df['subcategory'].value_counts())

    print("\nDish types:")
    print(df['dish_type'].value_counts())

analyze_recipe_data(cleaned_df)

def find_insights(df):
    # Most common ingredients
    all_ingredients = []
    for ingredients in df['ingredients']:
        all_ingredients.extend([ing.lower().strip() for ing in ingredients])

    print("\nTop 20 most common ingredients:")
    print(Counter(all_ingredients).most_common(20))

    # Recipes by author
    print("\nMost prolific authors:")
    print(df['author'].value_counts().head(10))

    # High protein recipes
    high_protein = df[df['nutrient_protein'] > 20].sort_values('nutrient_protein', ascending=False)
    print("\nHighest protein recipes:")
    print(high_protein[['name', 'nutrient_protein']].head(5))

    # Quick recipes (under 15 mins prep)
    quick_recipes = df[df['prep_time_mins'] <= 15]
    print(f"\nNumber of quick recipes (≤15 min prep): {len(quick_recipes)}")

find_insights(cleaned_df)

# Save cleaned data
cleaned_df.to_csv('cleaned_recipes.csv', index=False)
cleaned_df.to_json('cleaned_recipes.json', orient='records')

"""# Training

## Structurum og openAI integration
"""

from openai import OpenAI
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Initialize the OpenAI client
# Option 1: Set environment variable (recommended for security)
# os.environ['OPENAI_API_KEY'] = "your-api-key"

# Option 2: Pass directly to client (less secure)
client = OpenAI(api_key="api key here")

# PART 1: DATA PREPARATION FUNCTIONS
First one is : from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


class RecipeGenerator:
    def __init__(self, embeddings_file=None, df=None, api_key=None):
        """
        Initialize the RecipeGenerator with recipe data and embeddings.

        Args:
            embeddings_file: Path to pickle file with recipe data and embeddings
            df: DataFrame with recipe data and embeddings (alternative to embeddings_file)
            api_key: OpenAI API key
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or "API here")

        # Load data
        if df is not None:
            self.df = df.dropna(subset=['embedding'])
        elif embeddings_file is not None:
            self.df = pd.read_pickle(embeddings_file).dropna(subset=['embedding'])
        else:
            raise ValueError("Either embeddings_file or df must be provided")

        # Extract embeddings
        self.embeddings = np.array(self.df['embedding'].tolist())
        print(f"Recipe generator initialized with {len(self.df)} recipes")

    def get_embedding(self, text, model="text-embedding-ada-002"):
        """Generate embedding for a text query."""
        if not text or not isinstance(text, str):
            return None

        text = text.replace("\n", " ").strip()[:8000]
        if not text:
            return None

        try:
            response = self.client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def find_similar_recipes(self, query, n=4):
        """
        Find recipes similar to the query.

        Args:
            query: Text query to find similar recipes
            n: Number of similar recipes to return

        Returns:
            DataFrame with similar recipes
        """
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            print("Error: Could not generate embedding for query")
            return pd.DataFrame()

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-n:][::-1]

        similar_recipes = self.df.iloc[top_indices].copy()
        similar_recipes['similarity'] = similarities[top_indices]

        print(f"Found {len(similar_recipes)} similar recipes for '{query}'")
        return similar_recipes

        
    
    def generate_new_recipe(self, theme, difficulty=None, cuisine=None, dietary=None):
        """
        Generate a new recipe based on theme and other parameters.

        Args:
            theme: Theme or main ingredient for the recipe
            difficulty: Difficulty level (optional)
            cuisine: Cuisine type (optional)
            dietary: Dietary restrictions (optional)

        Returns:
            Generated recipe text
        """
        # Find similar recipes for context
        similar_recipes = self.find_similar_recipes(theme)

        if len(similar_recipes) == 0:
            print("Warning: No similar recipes found for context")

        # Build the prompt
        prompt = f"""I want to create a new recipe with the following characteristics:
Theme: {theme}
Difficulty: {difficulty or 'Any'}
Cuisine: {cuisine or 'Any'}
Dietary: {dietary or 'None'}

Here are some similar recipes for inspiration:
"""

        for i, (idx, recipe) in enumerate(similar_recipes.iterrows()):
            prompt += f"\n--- Recipe {i+1} ---\n"
            prompt += f"Name: {recipe.get('name', 'Unknown')}\n"

            if 'description' in recipe:
                prompt += f"Description: {recipe['description']}\n"

            # Handle ingredients in different formats
            if 'ingredients' in recipe:
                ingredients = recipe['ingredients']
                if isinstance(ingredients, str):
                    try:
                        ingredients_list = eval(ingredients)
                        if isinstance(ingredients_list, list):
                            prompt += f"Main Ingredients: {', '.join(ingredients_list[:5])}\n"
                        else:
                            prompt += f"Main Ingredients: {ingredients}\n"
                    except:
                        prompt += f"Main Ingredients: {ingredients}\n"
                elif isinstance(ingredients, list):
                    prompt += f"Main Ingredients: {', '.join(ingredients[:5])}\n"

        prompt += """
Please generate a completely new recipe that follows similar patterns but is original.
Include:
1. A creative recipe name
2. A brief description
3. Preparation and cooking times
4. Number of servings
5. Detailed ingredients list
6. Step-by-step instructions
7. Estimated nutritional information

The recipe should be practical to make at home with common ingredients.
"""

        # Generate the recipe using OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # You can use "gpt-4" for better results if available
                messages=[
                    {"role": "system", "content": "You are a professional chef who creates practical, delicious recipes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating recipe: {e}")
            return f"Error generating recipe: {e}"

      

# Example usage
if __name__ == "__main__":
    # Example:
     generator = RecipeGenerator(
         embeddings_file='recipes_with_embeddings.pkl',
         api_key='API key here'
         )
     new_recipe = generator.generate_new_recipe(
         theme="pasta with meatballs and tomato sauce",
         difficulty="Easy",
         cuisine="Italian"First one is : from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


class RecipeGenerator:
    def __init__(self, embeddings_file=None, df=None, api_key=None):
        """
        Initialize the RecipeGenerator with recipe data and embeddings.

        Args:
            embeddings_file: Path to pickle file with recipe data and embeddings
            df: DataFrame with recipe data and embeddings (alternative to embeddings_file)
            api_key: OpenAI API key
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or "API key here")

        # Load data
        if df is not None:
            self.df = df.dropna(subset=['embedding'])
        elif embeddings_file is not None:
            self.df = pd.read_pickle(embeddings_file).dropna(subset=['embedding'])
        else:
            raise ValueError("Either embeddings_file or df must be provided")

        # Extract embeddings
        self.embeddings = np.array(self.df['embedding'].tolist())
        print(f"Recipe generator initialized with {len(self.df)} recipes")

    def get_embedding(self, text, model="text-embedding-ada-002"):
        """Generate embedding for a text query."""
        if not text or not isinstance(text, str):
            return None

        text = text.replace("\n", " ").strip()[:8000]
        if not text:
            return None

        try:
            response = self.client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def find_similar_recipes(self, query, n=4):
        """
        Find recipes similar to the query.

        Args:
            query: Text query to find similar recipes
            n: Number of similar recipes to return

        Returns:
            DataFrame with similar recipes
        """
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            print("Error: Could not generate embedding for query")
            return pd.DataFrame()

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-n:][::-1]

        similar_recipes = self.df.iloc[top_indices].copy()
        similar_recipes['similarity'] = similarities[top_indices]

        print(f"Found {len(similar_recipes)} similar recipes for '{query}'")
        return similar_recipes

        
    
    def generate_new_recipe(self, theme, difficulty=None, cuisine=None, dietary=None):
        """
        Generate a new recipe based on theme and other parameters.

        Args:
            theme: Theme or main ingredient for the recipe
            difficulty: Difficulty level (optional)
            cuisine: Cuisine type (optional)
            dietary: Dietary restrictions (optional)

        Returns:
            Generated recipe text
        """
        # Find similar recipes for context
        similar_recipes = self.find_similar_recipes(theme)

        if len(similar_recipes) == 0:
            print("Warning: No similar recipes found for context")

        # Build the prompt
        prompt = f"""I want to create a new recipe with the following characteristics:
Theme: {theme}
Difficulty: {difficulty or 'Any'}
Cuisine: {cuisine or 'Any'}
Dietary: {dietary or 'None'}

Here are some similar recipes for inspiration:
"""

        for i, (idx, recipe) in enumerate(similar_recipes.iterrows()):
            prompt += f"\n--- Recipe {i+1} ---\n"
            prompt += f"Name: {recipe.get('name', 'Unknown')}\n"

            if 'description' in recipe:
                prompt += f"Description: {recipe['description']}\n"

            # Handle ingredients in different formats
            if 'ingredients' in recipe:
                ingredients = recipe['ingredients']
                if isinstance(ingredients, str):
                    try:
                        ingredients_list = eval(ingredients)
                        if isinstance(ingredients_list, list):
                            prompt += f"Main Ingredients: {', '.join(ingredients_list[:5])}\n"
                        else:
                            prompt += f"Main Ingredients: {ingredients}\n"
                    except:
                        prompt += f"Main Ingredients: {ingredients}\n"
                elif isinstance(ingredients, list):
                    prompt += f"Main Ingredients: {', '.join(ingredients[:5])}\n"

        prompt += """
Please generate a completely new recipe that follows similar patterns but is original.
Include:
1. A creative recipe name
2. A brief description
3. Preparation and cooking times
4. Number of servings
5. Detailed ingredients list
6. Step-by-step instructions
7. Estimated nutritional information

The recipe should be practical to make at home with common ingredients.
"""

        # Generate the recipe using OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # You can use "gpt-4" for better results if available
                messages=[
                    {"role": "system", "content": "You are a professional chef who creates practical, delicious recipes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating recipe: {e}")
            return f"Error generating recipe: {e}"

      

# Example usage
if __name__ == "__main__":
    # Example:
     generator = RecipeGenerator(
         embeddings_file='recipes_with_embeddings.pkl',
         api_key='API key here'
         )
     new_recipe = generator.generate_new_recipe(
         theme="pasta with meatballs and tomato sauce",
         difficulty="Easy",
         cuisine="Italian"
     )
     print(new_recipe)

     print("Recipe generator loaded. Import and use the class as needed.")
     )
     print(new_recipe)

     print("Recipe generator loaded. Import and use the class as needed.")
def create_recipe_text(row):
    """
    Create a text representation of a recipe from structured data.

    Args:
        row: A pandas DataFrame row containing recipe data

    Returns:
        str: Formatted recipe text
    """
    try:
        # Handle different formats of ingredients and steps (string or list)
        if isinstance(row['ingredients'], str):
            try:
                ingredients = eval(row['ingredients'])
                ingredients_text = "\n".join([f"- {ing}" for ing in ingredients])
            except:
                ingredients_text = row['ingredients']
        else:
            ingredients_text = "\n".join([f"- {ing}" for ing in row['ingredients']])

        if isinstance(row['steps'], str):
            try:
                steps = eval(row['steps'])
                steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
            except:
                steps_text = row['steps']
        else:
            steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(row['steps'])])

        # Create the formatted recipe text
        recipe_text = f"""
Recipe Name: {row.get('name', 'Unknown')}
Description: {row.get('description', '')}
Difficulty: {row.get('difficult', row.get('difficulty', 'Unknown'))}
Preparation Time: {row.get('prep_time', 'Unknown')}
Cooking Time: {row.get('cook_time', 'Unknown')}
Servings: {row.get('serves', row.get('servings', 'Unknown'))}

Ingredients:
{ingredients_text}

Instructions:
{steps_text}
"""

        # Add nutritional information if available
        if any(col in row for col in ['nutrient_kcal', 'nutrient_protein', 'nutrient_carbs', 'nutrient_fat']):
            nutrition_text = f"""
Nutritional Information (per serving):
- Calories: {row.get('nutrient_kcal', 'Unknown')} kcal
- Protein: {row.get('nutrient_protein', 'Unknown')}g
- Carbohydrates: {row.get('nutrient_carbs', 'Unknown')}g
- Fat: {row.get('nutrient_fat', 'Unknown')}g
"""
            recipe_text += nutrition_text

        return recipe_text

    except Exception as e:
        print(f"Error creating recipe text: {e}")
        # Return a simplified version if there's an error
        return f"Recipe: {row.get('name', 'Unknown')}\nIngredients: {row.get('ingredients', '')}\nSteps: {row.get('steps', '')}"

def prepare_recipe_data(input_file, train_output='train_recipes.csv', test_output='test_recipes.csv', test_size=0.2, random_state=42):
    """
    Prepare recipe data by creating text representations and splitting into train/test sets.

    Args:
        input_file: Path to the input CSV file with recipe data
        train_output: Path to save the training set
        test_output: Path to save the test set
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        tuple: (train_df, test_df) DataFrames with prepared data
    """
    try:
        # Load data
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)

        # Create text representation
        print("Creating recipe text representations...")
        df['recipe_text'] = df.apply(create_recipe_text, axis=1)

        # Split into training and test sets
        print(f"Splitting data into train ({1-test_size:.0%}) and test ({test_size:.0%}) sets...")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        # Save the splits
        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)

        print(f"Saved train data to {train_output} ({len(train_df)} recipes)")
        print(f"Saved test data to {test_output} ({len(test_df)} recipes)")

        return train_df, test_df

    except Exception as e:
        print(f"Error preparing recipe data: {e}")
        return None, None

# PART 2: EMBEDDING GENERATION FUNCTIONS

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generate embedding for a single text using the OpenAI API.

    Args:
        text: Text to generate embedding for
        model: OpenAI embedding model to use

    Returns:
        list: Embedding vector or None if error
    """
    if not text or not isinstance(text, str):
        print("Error: Invalid text input")
        return None

    text = text.replace("\n", " ").strip()
    if not text:
        print("Error: Empty text after cleaning")
        return None

    # Truncate to 8000 characters as per OpenAI's limit
    text = text[:8000]

    try:
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def generate_embeddings_batch(texts, batch_size=20, delay=1):
    """
    Generate embeddings for a batch of texts to improve efficiency.

    Args:
        texts: List of texts to generate embeddings for
        batch_size: Number of texts to process in each batch
        delay: Delay between batches in seconds

    Returns:
        list: List of embedding vectors
    """
    embeddings = []

    # Filter out invalid texts
    valid_texts = []
    valid_indices = []

    for i, text in enumerate(texts):
        if text and isinstance(text, str):
            cleaned_text = text.replace("\n", " ").strip()[:8000]
            if cleaned_text:
                valid_texts.append(cleaned_text)
                valid_indices.append(i)

    # Process valid texts in batches
    for i in tqdm(range(0, len(valid_texts), batch_size)):
        batch = valid_texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [data.embedding for data in response.data]

            # Map embeddings back to their original positions
            for j, emb in enumerate(batch_embeddings):
                orig_idx = valid_indices[i+j]
                while len(embeddings) <= orig_idx:
                    embeddings.append(None)
                embeddings[orig_idx] = emb

            # Add delay to avoid rate limiting
            time.sleep(delay)
        except Exception as e:
            print(f"Error on batch {i//batch_size}: {e}")
            # Add None for each failed embedding
            for j in range(len(batch)):
                orig_idx = valid_indices[i+j]
                while len(embeddings) <= orig_idx:
                    embeddings.append(None)
                embeddings[orig_idx] = None

    # Fill any remaining positions with None
    while len(embeddings) < len(texts):
        embeddings.append(None)

    return embeddings

def generate_embeddings(df, output_file='recipes_with_embeddings.pkl', batch_size=20, delay=1):
    """
    Generate embeddings for recipes in a DataFrame and save to file.

    Args:
        df: DataFrame containing recipe data with 'recipe_text' column
        output_file: Path to save the DataFrame with embeddings
        batch_size: Number of texts to process in each batch
        delay: Delay between batches in seconds

    Returns:
        DataFrame: DataFrame with embeddings added
    """
    if 'recipe_text' not in df.columns:
        print("Error: DataFrame doesn't have 'recipe_text' column")
        return df

    print(f"Generating embeddings for {len(df)} recipes...")

    # Generate embeddings
    texts = df['recipe_text'].fillna('').tolist()
    df['embedding'] = generate_embeddings_batch(texts, batch_size=batch_size, delay=delay)

    # Count valid embeddings
    valid_count = df['embedding'].count()
    print(f"Generated {valid_count} valid embeddings out of {len(df)} recipes")

    # Save embeddings
    df.to_pickle(output_file)
    print(f"Saved embeddings to {output_file}")

    return df

# PART 3: MAIN FUNCTION TO RUN THE COMPLETE PIPELINE

def run_recipe_pipeline(input_file, embeddings_output='recipes_with_embeddings.pkl',
                       train_output='train_recipes.csv', test_output='test_recipes.csv',
                       test_size=0.2, batch_size=20, delay=1):
    """
    Run the complete recipe pipeline: data preparation and embedding generation.

    Args:
        input_file: Path to the input CSV file with recipe data
        embeddings_output: Path to save the DataFrame with embeddings
        train_output: Path to save the training set
        test_output: Path to save the test set
        test_size: Proportion of data to use for testing
        batch_size: Number of texts to process in each batch for embeddings
        delay: Delay between batches in seconds

    Returns:
        tuple: (train_df, test_df, embeddings_df) DataFrames with prepared data and embeddings
    """
    # Step 1: Prepare data
    train_df, test_df = prepare_recipe_data(
        input_file=input_file,
        train_output=train_output,
        test_output=test_output,
        test_size=test_size
    )

    if train_df is None or test_df is None:
        print("Error in data preparation. Stopping pipeline.")
        return None, None, None

    # Step 2: Generate embeddings for all data
    all_df = pd.concat([train_df, test_df])
    embeddings_df = generate_embeddings(
        df=all_df,
        output_file=embeddings_output,
        batch_size=batch_size,
        delay=delay
    )

    return train_df, test_df, embeddings_df

# Example usage
if __name__ == "__main__":
    # To run the complete pipeline:
    train_df, test_df, embeddings_df = run_recipe_pipeline('cleaned_recipes.csv')

    # Or to run just the data preparation:
    # train_df, test_df = prepare_recipe_data('cleaned_recipes.csv')

    # Or to run just the embedding generation:
    # df = pd.read_csv('cleaned_recipes.csv')
    # df['recipe_text'] = df.apply(create_recipe_text, axis=1)
    # embeddings_df = generate_embeddings(df)

    print("Recipe generator module loaded. Import and use the functions as needed.")

"""## Eiithvað dót frá manus til að visualiza embedding"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.patches as mpatches

def visualize_recipe_embeddings_by_category(embeddings_file, category_column='category', output_image='recipe_categories_visualization.png'):
    """
    Visualize recipe embeddings using t-SNE with colored circles around points in the same category.

    Args:
        embeddings_file: Path to the pickle file containing recipe embeddings
        category_column: Column name in the DataFrame that contains category information
        output_image: Path to save the visualization image

    Returns:
        None (saves visualization to file)
    """
    # Load embeddings
    print(f"Loading embeddings from {embeddings_file}...")
    df = pd.read_pickle(embeddings_file)

    # Filter out recipes without embeddings
    df_with_embeddings = df[df['embedding'].notna()].copy()
    print(f"Found {len(df_with_embeddings)} recipes with valid embeddings")

    if len(df_with_embeddings) < 2:
        print("Not enough recipes with embeddings for visualization")
        return

    # Check if category column exists
    if category_column not in df_with_embeddings.columns:
        print(f"Warning: '{category_column}' column not found. Using recipe names only.")
        has_categories = False
    else:
        has_categories = True

    # Extract embeddings into a numpy array
    embeddings_array = np.array(df_with_embeddings['embedding'].tolist())

    # Use t-SNE to reduce dimensions to 2D for visualization
    print("Reducing dimensions with t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_with_embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings_array)

    # Create a scatter plot
    plt.figure(figsize=(12, 10))

    # If we have categories, use them for coloring
    if has_categories:
        # Get unique categories and assign colors
        categories = df_with_embeddings[category_column].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))

        # Create a color map
        category_color_map = dict(zip(categories, colors))

        # Plot each category with its own color and add circles around groups
        for category, color in category_color_map.items():
            # Get indices for this category
            category_indices = df_with_embeddings[df_with_embeddings[category_column] == category].index

            # Get points for this category
            category_points = embeddings_2d[[df_with_embeddings.index.get_loc(idx) for idx in category_indices]]

            # Plot points for this category
            plt.scatter(category_points[:, 0], category_points[:, 1],
                       color=color, alpha=0.7, label=category)

            # If we have more than one point in this category, draw a circle around the group
            if len(category_points) > 1:
                # Find the center of the points
                center_x = np.mean(category_points[:, 0])
                center_y = np.mean(category_points[:, 1])

                # Calculate radius to encompass all points plus some padding
                distances = np.sqrt(((category_points[:, 0] - center_x) ** 2) +
                                   ((category_points[:, 1] - center_y) ** 2))
                radius = np.max(distances) * 1.2  # Add 20% padding

                # Draw the circle
                circle = plt.Circle((center_x, center_y), radius, color=color,
                                   fill=False, alpha=0.5, linewidth=2)
                plt.gca().add_patch(circle)

        # Add a legend
        plt.legend(loc='best')
    else:
        # If no categories, just plot all points in one color
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

    # Add recipe names as labels (if available)
    if 'name' in df_with_embeddings.columns:
        for i, (idx, row) in enumerate(df_with_embeddings.iterrows()):
            plt.annotate(row['name'], (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                         fontsize=8, alpha=0.8)

    plt.title('Recipe Embeddings Visualization by Category')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()

    # Save the visualization
    plt.savefig(output_image, dpi=300)
    print(f"Visualization saved to {output_image}")

    # Close the plot to free memory
    plt.close()

# Example usage
if __name__ == "__main__":
    # Example: Add a category column to your DataFrame before saving embeddings
    # df['category'] = df['name'].apply(lambda x: 'Dessert' if 'cake' in x.lower() or 'cookie' in x.lower() else
    #                                          ('Pasta' if 'pasta' in x.lower() or 'spaghetti' in x.lower() else 'Other'))
    # df.to_pickle('recipes_with_categories.pkl')

    # Then visualize with categories
    # visualize_recipe_embeddings_by_category('recipes_with_categories.pkl', category_column='category')

    print("Enhanced visualization utility loaded. Import and use the function as needed.")



def find_similar_recipes(embeddings_file, recipe_index=0, top_n=5):
    """
    Find similar recipes based on embedding similarity.

    Args:
        embeddings_file: Path to the pickle file containing recipe embeddings
        recipe_index: Index of the recipe to find similar recipes for
        top_n: Number of similar recipes to return

    Returns:
        DataFrame with similar recipes
    """
    # Load embeddings
    df = pd.read_pickle(embeddings_file)

    # Filter out recipes without embeddings
    df_with_embeddings = df[df['embedding'].notna()].copy()

    if len(df_with_embeddings) < 2:
        print("Not enough recipes with embeddings for similarity comparison")
        return None

    # Get the target recipe embedding
    target_embedding = df_with_embeddings.iloc[recipe_index]['embedding']
    target_name = df_with_embeddings.iloc[recipe_index].get('name', f"Recipe {recipe_index}")

    # Extract all embeddings
    all_embeddings = np.array(df_with_embeddings['embedding'].tolist())

    # Calculate cosine similarity
    similarities = cosine_similarity([target_embedding], all_embeddings)[0]

    # Add similarity scores to the DataFrame
    df_with_embeddings['similarity'] = similarities

    # Sort by similarity (excluding the target recipe itself)
    similar_recipes = df_with_embeddings.sort_values('similarity', ascending=False).iloc[1:top_n+1]

    print(f"Top {top_n} recipes similar to '{target_name}':")
    for i, (idx, row) in enumerate(similar_recipes.iterrows()):
        name = row.get('name', f"Recipe {idx}")
        similarity = row['similarity'] * 100  # Convert to percentage
        print(f"{i+1}. {name} (Similarity: {similarity:.2f}%)")

    return similar_recipes[['name', 'similarity', 'recipe_text']]

def search_recipes_by_query(embeddings_file, query_text, top_n=5, client=None):
    """
    Search for recipes using a text query by comparing embeddings.

    Args:
        embeddings_file: Path to the pickle file containing recipe embeddings
        query_text: Text query to search for
        top_n: Number of similar recipes to return
        client: OpenAI client (if None, will create a new one)

    Returns:
        DataFrame with matching recipes
    """
    # Import here to avoid circular imports
    from openai import OpenAI

    # Load embeddings
    df = pd.read_pickle(embeddings_file)

    # Filter out recipes without embeddings
    df_with_embeddings = df[df['embedding'].notna()].copy()

    if len(df_with_embeddings) < 1:
        print("No recipes with embeddings found")
        return None

    # Create OpenAI client if not provided
    if client is None:
        client = OpenAI(api_key="APi key here")

    # Generate embedding for the query
    try:
        response = client.embeddings.create(
            input=[query_text],
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

    # Extract all recipe embeddings
    all_embeddings = np.array(df_with_embeddings['embedding'].tolist())

    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]

    # Add similarity scores to the DataFrame
    df_with_embeddings['similarity'] = similarities

    # Sort by similarity
    matching_recipes = df_with_embeddings.sort_values('similarity', ascending=False).head(top_n)

    print(f"Top {top_n} recipes matching query '{query_text}':")
    for i, (idx, row) in enumerate(matching_recipes.iterrows()):
        name = row.get('name', f"Recipe {idx}")
        similarity = row['similarity'] * 100  # Convert to percentage
        print(f"{i+1}. {name} (Match: {similarity:.2f}%)")

    return matching_recipes[['name', 'similarity', 'recipe_text']]

# Example usage
if __name__ == "__main__":
    # To visualize recipe embeddings:
    visualize_recipe_embeddings_by_category('recipes_with_embeddings.pkl')

    # To find similar recipes:
    # similar_recipes = find_similar_recipes('recipes_with_embeddings.pkl', recipe_index=0, top_n=5)

    # To search recipes by query:
    # matching_recipes = search_recipes_by_query('recipes_with_embeddings.pkl', 'quick vegetarian dinner', top_n=5)

    print("Recipe embedding utilities loaded. Import and use the functions as needed.")

"""##  Build the Recipe Generation System"""

#RecipeGenerator
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


class RecipeGenerator:
    def __init__(self, embeddings_file=None, df=None, api_key=None):
        """
        Initialize the RecipeGenerator with recipe data and embeddings.

        Args:
            embeddings_file: Path to pickle file with recipe data and embeddings
            df: DataFrame with recipe data and embeddings (alternative to embeddings_file)
            api_key: OpenAI API key
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or "API key here")

        # Load data
        if df is not None:
            self.df = df.dropna(subset=['embedding'])
        elif embeddings_file is not None:
            self.df = pd.read_pickle(embeddings_file).dropna(subset=['embedding'])
        else:
            raise ValueError("Either embeddings_file or df must be provided")

        # Extract embeddings
        self.embeddings = np.array(self.df['embedding'].tolist())
        print(f"Recipe generator initialized with {len(self.df)} recipes")

    def get_embedding(self, text, model="text-embedding-ada-002"):
        """Generate embedding for a text query."""
        if not text or not isinstance(text, str):
            return None

        text = text.replace("\n", " ").strip()[:8000]
        if not text:
            return None

        try:
            response = self.client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def find_similar_recipes(self, query, n=4):
        """
        Find recipes similar to the query.

        Args:
            query: Text query to find similar recipes
            n: Number of similar recipes to return

        Returns:
            DataFrame with similar recipes
        """
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            print("Error: Could not generate embedding for query")
            return pd.DataFrame()

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-n:][::-1]

        similar_recipes = self.df.iloc[top_indices].copy()
        similar_recipes['similarity'] = similarities[top_indices]

        print(f"Found {len(similar_recipes)} similar recipes for '{query}'")
        return similar_recipes



    def generate_new_recipe(self, theme, difficulty=None, cuisine=None, dietary=None):
        """
        Generate a new recipe based on theme and other parameters.

        Args:
            theme: Theme or main ingredient for the recipe
            difficulty: Difficulty level (optional)
            cuisine: Cuisine type (optional)
            dietary: Dietary restrictions (optional)

        Returns:
            Generated recipe text
        """
        # Find similar recipes for context
        similar_recipes = self.find_similar_recipes(theme)

        if len(similar_recipes) == 0:
            print("Warning: No similar recipes found for context")

        # Build the prompt
        prompt = f"""I want to create a new recipe with the following characteristics:
Theme: {theme}
Difficulty: {difficulty or 'Any'}
Cuisine: {cuisine or 'Any'}
Dietary: {dietary or 'None'}

Here are some similar recipes for inspiration:
"""

        for i, (idx, recipe) in enumerate(similar_recipes.iterrows()):
            prompt += f"\n--- Recipe {i+1} ---\n"
            prompt += f"Name: {recipe.get('name', 'Unknown')}\n"

            if 'description' in recipe:
                prompt += f"Description: {recipe['description']}\n"

            # Handle ingredients in different formats
            if 'ingredients' in recipe:
                ingredients = recipe['ingredients']
                if isinstance(ingredients, str):
                    try:
                        ingredients_list = eval(ingredients)
                        if isinstance(ingredients_list, list):
                            prompt += f"Main Ingredients: {', '.join(ingredients_list[:5])}\n"
                        else:
                            prompt += f"Main Ingredients: {ingredients}\n"
                    except:
                        prompt += f"Main Ingredients: {ingredients}\n"
                elif isinstance(ingredients, list):
                    prompt += f"Main Ingredients: {', '.join(ingredients[:5])}\n"

        prompt += """
Please generate a completely new recipe that follows similar patterns but is original.
Include:
1. A creative recipe name
2. A brief description
3. Preparation and cooking times
4. Number of servings
5. Detailed ingredients list
6. Step-by-step instructions
7. Estimated nutritional information

The recipe should be practical to make at home with common ingredients.
"""

        # Generate the recipe using OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # You can use "gpt-4" for better results if available
                messages=[
                    {"role": "system", "content": "You are a professional chef who creates practical, delicious recipes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating recipe: {e}")
            return f"Error generating recipe: {e}"



# Example usage
if __name__ == "__main__":
    # Example:
     generator = RecipeGenerator(
         embeddings_file='recipes_with_embeddings.pkl',
         api_key='API here'
         )
     new_recipe = generator.generate_new_recipe(
         theme="pasta with meatballs and tomato sauce",
         difficulty="Easy",
         cuisine="Italian"
     )
     print(new_recipe)

     print("Recipe generator loaded. Import and use the class as needed.")

#RecipeGenerator kóði með mynd með

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re  # Added for regex

class RecipeGenerator:
    def __init__(self, embeddings_file=None, df=None, api_key=None):
        self.client = OpenAI(api_key=api_key)

        if df is not None:
            self.df = df.dropna(subset=['embedding'])
        elif embeddings_file is not None:
            self.df = pd.read_pickle(embeddings_file).dropna(subset=['embedding'])
        else:
            raise ValueError("Either embeddings_file or df must be provided")

        self.embeddings = np.array(self.df['embedding'].tolist())
        print(f"Recipe generator initialized with {len(self.df)} recipes")

    def get_embedding(self, text, model="text-embedding-ada-002"):
        if not text or not isinstance(text, str):
            return None

        text = text.replace("\n", " ").strip()[:8000]
        if not text:
            return None

        try:
            response = self.client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def find_similar_recipes(self, query, n=4):
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            print("Error: Could not generate embedding for query")
            return pd.DataFrame()

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-n:][::-1]

        similar_recipes = self.df.iloc[top_indices].copy()
        similar_recipes['similarity'] = similarities[top_indices]

        print(f"Found {len(similar_recipes)} similar recipes for '{query}'")
        return similar_recipes

    def extract_recipe_name(self, text):
        match = re.search(r'(?im)^name\s*[:\-]\s*(.+)', text)
        if match:
            return match.group(1).strip()
        return None

    def generate_image_for_recipe(self, recipe_name):
        try:
            image_prompt = f"A realistic, professional photo of a dish called '{recipe_name}'. Food photography style."
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            return response.data[0].url
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def generate_new_recipe(self, theme, difficulty=None, cuisine=None, dietary=None):
        similar_recipes = self.find_similar_recipes(theme)

        if len(similar_recipes) == 0:
            print("Warning: No similar recipes found for context")

        prompt = f"""I want to create a new recipe with the following characteristics:
Theme: {theme}
Difficulty: {difficulty or 'Any'}
Cuisine: {cuisine or 'Any'}
Dietary: {dietary or 'None'}

Here are some similar recipes for inspiration:
"""

        for i, (idx, recipe) in enumerate(similar_recipes.iterrows()):
            prompt += f"\n--- Recipe {i+1} ---\n"
            prompt += f"Name: {recipe.get('name', 'Unknown')}\n"

            if 'description' in recipe:
                prompt += f"Description: {recipe['description']}\n"

            if 'ingredients' in recipe:
                ingredients = recipe['ingredients']
                if isinstance(ingredients, str):
                    try:
                        ingredients_list = eval(ingredients)
                        if isinstance(ingredients_list, list):
                            prompt += f"Main Ingredients: {', '.join(ingredients_list[:5])}\n"
                        else:
                            prompt += f"Main Ingredients: {ingredients}\n"
                    except:
                        prompt += f"Main Ingredients: {ingredients}\n"
                elif isinstance(ingredients, list):
                    prompt += f"Main Ingredients: {', '.join(ingredients[:5])}\n"

        prompt += """
Please generate a completely new recipe that follows similar patterns but is original.
Include:
1. A creative recipe name
2. A brief description
3. Preparation and cooking times
4. Number of servings
5. Detailed ingredients list
6. Step-by-step instructions
7. Estimated nutritional information

The recipe should be practical to make at home with common ingredients.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional chef who creates practical, delicious recipes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            recipe_text = response.choices[0].message.content
            recipe_name = self.extract_recipe_name(recipe_text)
            image_url = self.generate_image_for_recipe(recipe_name) if recipe_name else None

            return {
                "name": recipe_name,
                "text": recipe_text,
                "image_url": image_url
            }

        except Exception as e:
            print(f"Error generating recipe: {e}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    generator = RecipeGenerator(
        embeddings_file='recipes_with_embeddings.pkl',
        api_key='API HERE'
    )

    result = generator.generate_new_recipe(
        theme=" tortilla with chicken and bananas",
        difficulty="medium",
    
        cuisine="mexican"
    )

    print("Recipe Name:", result.get("name"))
    print("Recipe Text:\n", result.get("text"))
    print("Image of the recipe (url):", result.get("image_url"))

new_recipe = generator.generate_new_recipe(
    theme="cake with anchovies and indian spices",
    difficulty="hard",
    #cuisine="Italian",
    dietary="Vegetarian"
)
print(new_recipe)
