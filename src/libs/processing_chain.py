import os
import pickle
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
        pickle.dump(model, open(f"{config.model_path}", "wb"))
        print(f"Model object was saved to: \n{config.model_path}")
    else:
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model object file not found at: \n{config.model_path}")
        # Load the data and train a new model
        print("Loading an existing model.") 
        model = pickle.load(open(f"{config.model_path}", "rb"))

    return model