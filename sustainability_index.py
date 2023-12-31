import re

import pandas as pd
import spacy
from collections import Counter

# Load the English language model from spaCy
nlp = spacy.load("en_core_web_sm")
sustainable_words = [
            'Environment', 'Green', 'Conservation', 'Renewable', 'Eco-friendly', 'Carbon footprint',
            'Biodiversity', 'Climate change', 'Sustainable development', 'Recycling', 'Energy efficiency',
            'Zero waste', 'Eco-conscious', 'Responsible consumption', 'Renewable energy',
            'Sustainability reporting', 'Circular economy', 'Ethical sourcing', 'Water conservation',
            'Organic', 'Eco-system', 'Solar power', 'Greenwashing', 'Fair trade', 'Eco-label',
            'Social responsibility', 'Green building', 'Sustainable agriculture', 'Natural resources',
            'Pollution control'
        ]
# Function to count occurrences of keywords in a text
def count_keywords(text):
    count = 0
    for keyword in sustainable_words:
        # Use case-insensitive search
        count += len(re.findall(fr'\b{re.escape(keyword)}\b', text, flags=re.IGNORECASE))
        if count>0:
            print(count)
    return count


# List of CSV files
csv_files = ["dataset/BookingHotel - Thailand.csv", "dataset/BookingHotel - Singapore.csv"]  # Add your file names
# Process each CSV file
for file_name in csv_files:
    # Load CSV file
    df = pd.read_csv(file_name, encoding='ISO-8859-1')

    # Check if "Review Text" column exists in the DataFrame
    if "Text" in df.columns:
        # List of sustainable words (you can use the provided list or your own)
        # Apply the function to the 'Text' column
        df['Sustainable Words Count'] = df['Text'].apply(count_keywords)

        # Display the DataFrame with the tokenized results
        print(df[['Name', 'Text', 'Sustainable Words Count']])
