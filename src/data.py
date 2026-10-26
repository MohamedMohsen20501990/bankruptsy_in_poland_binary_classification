from dataclasses import dataclass
import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.logging_config import logger
from sklearn.model_selection import train_test_split
from src.preprocess import Preprocessor
from src.train import ModelTrainer



@dataclass
class DataConfig:
    """Store paths for the training and test datasets."""

    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path:str = os.path.join("artifacts", "test_data.csv")

class DataHandler:
    """Handle dataset loading, splitting, and saving."""

    def __init__(self):
        self.data_config = DataConfig()
        
    def read_csv_data(self, path:str):
        """Load a CSV file and validate required column types."""
        try:
            df = pd.read_csv(path, low_memory=False,na_values=["?"])
            df[["Class","Year"]]=df[["Class","Year"]].astype("int")
            df = df.reset_index(drop=True)
            df.drop(columns=["id"], inplace=True)
            return df
            
        except Exception:
            logger.exception("Error reading data from %s", path)
            raise
        
    def split_and_save(self, data_frame:pd.DataFrame):
        """Split the dataset and save training and test CSV files."""

        
        try:
            os.makedirs(os.path.dirname(self.data_config.train_data_path), exist_ok=True)
            train_data, test_data = train_test_split(data_frame, test_size=0.2, random_state=42, stratify=data_frame["Class"])
            train_data.to_csv(self.data_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.data_config.test_data_path, index=False, header=True)
            return train_data, test_data
            
        except Exception:
            logger.exception("Error splitting and saving data")
            raise    

if __name__== "__main__":
    data_handler= DataHandler()
    df = data_handler.read_csv_data("./data/raw/Bankruptcy Master Dataset.csv")     
    train_data, test_data = data_handler.split_and_save(df)
    
    # data_preprocessor = Preprocessor()
    # transformed_train_df, transformed_test_df, preprocessor_path =data_preprocessor.start_data_preprocessing(train_path=data_handler.data_config.train_data_path, test_path=data_handler.data_config.test_data_path)
    
    model_trainer = ModelTrainer()
    model_trainer.train_model(train_data=train_data, test_data=test_data, target="Class")
    print(model_trainer.report)