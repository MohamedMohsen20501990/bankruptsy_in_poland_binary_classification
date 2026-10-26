import pickle
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))


def save_obj(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        print(f"File save error: {e}")
        raise        

def load_obj(file_path):
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        return obj

    except Exception as e:
        print(f"File load error: {e}")
        raise    
    
def generate_input_features(rown_index:int = 1, year:float = 1.0):
    """Generates inputs from the origjnal DataFrame"""
    
    df = pd.read_csv("data/raw/Bankruptcy Master Dataset.csv",low_memory=False, index_col="id", dtype="float", na_values="?")
    df[["Class","Year"]]=df[["Class","Year"]].astype("int")
    df.index=df.index.astype("int")
    df = df.reset_index(drop=True)
    
    values = df.iloc[rown_index][[f"X{i}" for i in range(1, 65)]].values
    features_names = [f"X{i}"for i in range(1,65)]
    final_features = {}
    for feature, value in zip(features_names, values):
        final_features[feature] = value
        

    user_input = final_features
    user_input["Year"] = year
    return user_input    
