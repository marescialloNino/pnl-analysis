import pandas as pd

# Define the path to your .pkl file
pkl_file_path = "okx_bot_data/data_50fut_30m.pkl"

# Define the output CSV file path
csv_file_path = "okx_bot_data/price_data_okx.csv"

try:
    # Step 1: Read the pickle file into a DataFrame
    df = pd.read_pickle(pkl_file_path)
    
    # Optional: Inspect the DataFrame
    print("DataFrame loaded from pickle:")
    print(df.head())  # Show the first 5 rows
    
    # Step 2: Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)  # index=False avoids writing row indices
    print(f"Successfully saved to {csv_file_path}")

except FileNotFoundError:
    print(f"Error: The file {pkl_file_path} was not found.")
except Exception as e:
    print(f"Error occurred: {str(e)}")