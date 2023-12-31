import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
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
        if count > 0:
            print(count)
    return count


# List of CSV files
csv_files = ["dataset/BookingHotel - Thailand.csv",
             "dataset/BookingHotel - Singapore.csv"]  # Add your file names
# Process each CSV file
for file_name in csv_files:
    # Load CSV file
    df = pd.read_csv(file_name, encoding='ISO-8859-1')

    # Check if "Review Text" column exists in the DataFrame
    if "Text" in df.columns:
        # List of sustainable words (you can use the provided list or your own)
        # Apply the function to the 'Text' column
        df['Sustainable Words Count'] = df['Text'].apply(count_keywords)
        df['Sustainability Level'] = df['Sustainability Level'].fillna(0)
        sustainability_mapping = {
            'Travel Sustainable Level 1': 1,
            'Travel Sustainable Level 2': 2,
            'Travel Sustainable Level 3': 3,
            'Travel Sustainable Level 3+': 3
        }
        df['Sustainability Rating'] = df['Sustainability Level'].map(sustainability_mapping)

        # Features (X) - Use 'Sustainable Words Count' column as a feature
        X = df['Sustainable Words Count'].values.reshape(-1,1)

        # Target (y) - 'Sustainability Rating' column
        y = df['Sustainability Rating'].fillna(0)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Multinomial Logistic Regression model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(X_train, y_train)

        # Predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluation metrics
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

