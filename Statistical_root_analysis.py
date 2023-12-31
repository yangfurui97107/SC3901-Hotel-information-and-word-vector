import pandas as pd
import statsmodels.api as sm

# List of CSV files
csv_files = ["dataset/BookingHotel - Thailand.csv", "dataset/BookingHotel - Singapore.csv"]  # Add your file names
# Process each CSV file
for file_name in csv_files:
    # Load CSV file
    df = pd.read_csv(file_name, encoding='ISO-8859-1')

    # Specify the dependent variable and categorical independent variables
    dependent_variable = 'Review Score'
    categorical_variables = ['Reviewer Country', 'Room Type', 'Traveller']

    # Convert categorical variables to dummy variables (one-hot encoding)
    df_dummies = pd.get_dummies(df[categorical_variables], drop_first=True)

    # Concatenate the dummy variables with the original DataFrame
    df = pd.concat([df, df_dummies], axis=1)

    # Add a constant term to the independent variables
    X = sm.add_constant(df[df_dummies.columns])
    X = X.to_numpy(dtype=float)
    y = df[dependent_variable].to_numpy(dtype=float)
    # Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(y, X).fit()

    # Print the summary of the regression
    print(model.summary())