import spacy
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# List of CSV files
csv_files = ["data/BookingHotel - Thailand.csv","data/BookingHotel - Singapore.csv"]  # Add your file names
# Process each CSV file
for file_name in csv_files:
    # Load CSV file
    df = pd.read_csv(file_name, encoding='ISO-8859-1')

    # Check if "Review Text" column exists in the DataFrame
    if "Text" in df.columns:
        # Apply spaCy to each review text and store the vectors in a new column
        df["Word Vectors"] = df["Text"].apply(lambda text: nlp(str(text)).vector)

        # Create a new DataFrame with only the "Word Vectors" column
        output_df = pd.DataFrame({"Word Vectors": df["Word Vectors"]})

        # Save the "Word Vectors" DataFrame to a new CSV file
        output_file_name = file_name.replace(".csv", "_word_vectors.csv")
        output_df.to_csv(output_file_name, index=False)

        print(f"Word vectors for {file_name} saved to {output_file_name}")
    else:
        print(f"Error: 'Review Text' column not found in {file_name}")
