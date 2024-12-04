import kagglehub
import shutil
import os
import pandas as pd
from dotenv import load_dotenv
from logger import get_logger
from sacred import Ingredient

load_dotenv()

_logs = get_logger(__name__)

data_ingredient = Ingredient('data_ingredient')

data_ingredient.logger = _logs

@data_ingredient.config
def cfg():
    raw_data_path = os.getenv("RAW_DATA_PATH")
    processed_data_path = os.getenv("PROCESSED_DATA_PATH")
    dataset_identifier = os.getenv("DISEASE_DATASET_IDENTIFIER")
    cache_base_path = os.path.expanduser(os.getenv("KAGGLE_CACHE_PATH"))
    training_file_name = os.getenv("TRAINING_FILE_NAME")
    testing_file_name = os.getenv("TESTING_FILE_NAME")


@data_ingredient.capture
def get_kaggle_data(dataset_identifier, cache_base_path, raw_data_path):
    '''Download data from kaggle.'''

    # Build the specific cache directory path for the dataset
    cache_path = os.path.join(
        cache_base_path,
        dataset_identifier.replace("/", os.sep),
    )

    # Clean up the cache directory if it exists
    if os.path.exists(cache_path):
        _logs.info(f"Removing existing cache directory: {cache_path}")
        shutil.rmtree(cache_path)

    # Download the dataset (downloads to the default location)
    path = kagglehub.dataset_download(dataset_identifier)

    _logs.info(f"Temp path to dataset files: {path}")

    # Ensure the destination directory exists
    os.makedirs(raw_data_path, exist_ok=True)

    # Move the downloaded dataset to the destination
    for filename in os.listdir(path):
        shutil.move(os.path.join(path, filename), os.path.join(raw_data_path, filename))

    _logs.info(f"Dataset moved to: {raw_data_path}")


@data_ingredient.capture
def process_data(raw_data_path, processed_data_path):

    # Get all CSV files in the directory
    csv_files = [file for file in os.listdir(raw_data_path) if file.endswith(".csv")]

    # Process each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(raw_data_path, csv_file)
        _logs.info(f"Processing file: {csv_file}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check and print column names
        _logs.info(f"Original column names: {df.columns.tolist()}")

        # Drop unecessary colums
        if "fluid_overload" in df.columns:
            df.drop(columns=["fluid_overload"], inplace=True)
        if "Unnamed: 133" in df.columns:
            df.drop(columns=["Unnamed: 133"], inplace=True)

        # Rename columns
        if "fluid_overload.1" in df.columns:
            df.rename(columns={"fluid_overload.1": "fluid_overload"}, inplace=True)
        if "toxic_look_(typhos)" in df.columns:
            df.rename(columns={"toxic_look_(typhos)": "toxic_look_typhos"}, inplace=True)

        # Check for invalid entries, rows with all zeroes.
        all_zero_per_row = (df.iloc[:, :-1] == 0).all(axis=1) 
        count_zero_rows = all_zero_per_row.sum()

        _logs.info(f"Number of rows with all zeros (excluding last column): {count_zero_rows}")

        # Remove rows where all columns (except the last) are zero
        df_cleaned = df.loc[all_zero_per_row.eq(False)]

        # print("DataFrame after removing rows with all zeros (excluding last column):")
        # print(df_cleaned)
        
        # Save the updated DataFrame to the processed directory
        df_cleaned.to_csv(os.path.join(processed_data_path, csv_file), index=False)

        # Re-check column names
        _logs.info(f"Updated column names: {df.columns.tolist()}")

    # Get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(processed_data_path) if f.endswith(".csv")]

    # Read all files and validate columns
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(processed_data_path, file)
        dataframes[file] = pd.read_csv(file_path)

    # Validate columns across all files
    first_file = list(dataframes.keys())[0]
    reference_columns = list(dataframes[first_file].columns)

    mismatch_found = False
    for file, df in dataframes.items():
        if list(df.columns) != reference_columns:
            mismatch_found = True
            _logs.info(f"Mismatch found in file: {file}")
            _logs.info(f"Expected columns: {reference_columns}")
            _logs.info(f"Actual columns: {list(df.columns)}\n")

    if not mismatch_found:
        _logs.info("All files have matching columns and order.")

@data_ingredient.capture
def validate_dataset(processed_data_path, training_file_name, testing_file_name):
    """
    Validates the dataset based on the following conditions:
    - Training.csv and Testing.csv must exist.
    - Columns 'fluid_overload.1' and 'toxic_look_(typhos)' must not exist.
    - Column 'Unnamed: 133' must not exist in either file.
    
    Returns:
        bool: True if all conditions are met, False otherwise.
    """
    
    # Check if files exist
    training_path = os.path.join(processed_data_path, training_file_name)
    testing_path = os.path.join(processed_data_path, testing_file_name)
    if not os.path.exists(training_path) or not os.path.exists(testing_path):
        _logs.info(f"No processed files exist.")
        return False
    
    # Load the files
    try:
        training_data = pd.read_csv(training_path)
        testing_data = pd.read_csv(testing_path)
    except Exception as e:
        _logs.info(f"Error reading files: {e}")
        return False

    # Check for the absence of unwanted columns
    unwanted_columns = ["fluid_overload.1", "toxic_look_(typhos)", "Unnamed: 133"]
    for col in unwanted_columns:
        if col in training_data.columns or col in testing_data.columns:
            _logs.info("Processed data validation failed")
            return False

    # If all checks pass
    _logs.info("Processed data validation passed")
    return True

@data_ingredient.capture
def get_training_data(processed_data_path, training_file_name):
    '''Loads data from a given location.'''
    _logs.info(f'Getting training data from {processed_data_path}')
    file_path = os.path.join(processed_data_path, training_file_name)
    df = pd.read_csv(file_path)
    return df

@data_ingredient.capture
def get_validation_data(processed_data_path, testing_file_name):
    '''Loads data from a given location.'''
    _logs.info(f'Getting testing data from {processed_data_path}')
    file_path = os.path.join(processed_data_path, testing_file_name)
    df = pd.read_csv(processed_data_path)
    return df

@data_ingredient.capture
def load_data():
    '''Load training data and return X, Y'''
    counter = 5

    while (counter > 0 and not validate_dataset()):
        print("Failed data validation, re-processing data")
        get_kaggle_data()
        process_data()
        counter -= 1
        print(counter)
        
    df = get_training_data()
    X = df.drop(columns = ['prognosis'])
    Y = df[['prognosis']]

    return X, Y

@data_ingredient.capture
def load_validation_data():
    '''Load testing data and return X, Y'''
    counter = 5

    while (counter > 0 and not validate_dataset()):
        print("Failed data validation, re-processing data")
        get_kaggle_data()
        process_data()
        counter -= 1
        print(counter)
        
    df = get_validation_data()
    X = df.drop(columns = ['prognosis'])
    Y = df[['prognosis']]

    return X, Y