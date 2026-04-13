import os
import pickle
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(BASE_DIR)

from src.libs.config_tools import Config_Manager
from src.libs import data_tools, model_tools

def processing_chain(config: Config_Manager):
    
    if config.mode == "train":
        print("Training a new model.") 
        data = data_tools.Data_Manager(config)
        model = model_tools.Model_Manager(data, config)
        model_output = model_tools.Model_Output(
                    data=data,
                    model=model,
        )   
        # Deleting the model attribute before saving to avoid issues with Keras models and pickling
        pickle.dump(model, open(f"{config.model_path}", "wb"))

        # Creating a Dataframe for the forecast reults
        pickle.dump(model_output, open(f"{config.model_out_path}", "wb"))

        # Saving the Keras model separately
        model.model.save(config.tf_model_path)

        print(f"Model outputs were saved to: \n{config.model_folder}")
    else:
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model object file not found at: \n{config.model_path}")
        # Load the data and train a new model
        print("Loading an existing model.") 
        model = pickle.load(open(f"{config.model_path}", "rb"))
        model_output = pickle.load(open(f"{config.model_out_path}", "rb"))
        
    return model, model_output