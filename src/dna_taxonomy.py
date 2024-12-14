# don't forget hard coded 20 in ProcessData 
from measures import Measures
import os
import json
import pandas as pd
from preprocess import  DataPreProcess
from cnn_model import CNNModel
HOME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



# Load configurations from the JSON file
with open("config.json", "r") as file:
    MODEL_CONFIGS = json.load(file)
    
# Helper function to configure pandas display settings
def configure_pandas():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_colwidth", None)
    
def run_model(data_file, dataset_name, config, hierarchy=False):
    print(f"\nRunning model for dataset: {dataset_name}")
    print("Step 1: Preprocessing data...")
    dp = DataPreProcess(data_file, dataset_name, config)

    print("Step 2: Initializing measures and model...")
    measures_file = config.get("measures_output_file", None)
    measures = Measures(dp, filename=measures_file)
    model = CNNModel(measures, config)

    print("Step 3: Training the model...")
    if hierarchy:
        model.train_heirarchy_cv(dp)
    else:
        model.train_test_cv(dp)

    print("Step 4: Writing measures to file...")
    measures.write_measures(hierarchy=hierarchy)

    print("Step 5: Displaying final results...")
    configure_pandas()
    print(measures)

    return measures

def train_model(model_name, hierarchy=False):
    """
    Trains a model using the specified configuration from JSON.

    Parameters:
    - model_name (str): The key corresponding to the model in MODEL_CONFIGS.
    - hierarchy (bool): Whether to train the model with a hierarchical approach.

    Returns:
    - Result of the `run_model` function.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}")

    # Get the configuration for the specified model
    config = MODEL_CONFIGS[model_name]
    
    # Derive dataset name from the model name
    dataset = f"{model_name.capitalize()}_refseq.csv"
    
    # Run the model
    return run_model(dataset, model_name.capitalize(), config, hierarchy=hierarchy)


# Save results to a DataFrame and export to a file
def save_results_to_dataframe(results, output_file):
    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return results_df

# Specific training functions
def train_tymovirales_model():
    return train_model("tymovirales")

def train_martellivirales_model():
    return train_model("martellivirales")

def train_alsuviricetes_model():
    return train_model("alsuviricetes")

def train_hierarchical_model_for_alsuviricetes():
    return train_model("alsuviricetes", hierarchy=True)


# Entry point
if __name__ == "__main__":
    
    # Uncomment specific models to train
    #results.append(train_martellivirales_model())
    
    #m = train_hierarchical_model_for_alsuviricetes()
    m = train_tymovirales_model()
    
    #results.append(train_alsuviricetes_model())

    # Save all results to a DataFrame
    #save_results_to_dataframe(results, "all_results.csv")
    


   




