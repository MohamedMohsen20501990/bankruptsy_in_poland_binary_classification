import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass
from src.logging_config import logger
from src.utils import save_obj

@dataclass
class PreprocessorConfig:
    """Store paths for preprocessing artifacts and datasets."""

    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    transformed_train_data_path: str = os.path.join("artifacts", "transformed_train_data.parquet")
    transformed_test_data_path: str = os.path.join("artifacts", "transformed_test_data.parquet")
    
class Preprocessor:
    """Handle feature imputation, scaling, and data saving."""

    def __init__(self):
        self.preprocessor_config = PreprocessorConfig()   
    
    @staticmethod   
    def build_preprocessor(numerical_features: list[str]) -> ColumnTransformer:
        """Build the numerical imputation and scaling pipeline."""
        try:
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', RobustScaler())
            ]) 
            
            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_features),
            ], remainder="passthrough", verbose_feature_names_out=False)
            
            return preprocessor
            
        except Exception:
            logger.exception("Error building the preprocessor")
            raise
        
    def start_data_preprocessing(self, train_path:Path, test_path:Path):
        """Transform and save training and test datasets."""
        try:
            train_set = pd.read_csv(train_path, na_values=["?"], low_memory=False)
            test_set = pd.read_csv(test_path, na_values=["?"], low_memory=False)
            numerical_features = [col for col in train_set.columns if col.startswith("X")]
            target = "Class"
            
            X_train = train_set.drop(columns=[target])
            y_train= train_set[target]
            
            X_test = test_set.drop(columns=[target])
            y_test= test_set[target]
            
            preprocessor_obj = self.build_preprocessor(numerical_features)
            
            transformed_train_data = preprocessor_obj.fit_transform(X_train)
            transformed_test_data = preprocessor_obj.transform(X_test)
            
            feature_names = preprocessor_obj.get_feature_names_out()
            
            # sampler = RandomOverSampler(random_state=42)
            
            # sampled_train_data, y_train_sampled = sampler.fit_resample(transformed_train_data, y_train)
            
            transformed_train_df = pd.DataFrame(transformed_train_data, columns=feature_names)
            transformed_train_df[target] = y_train.values
            
            transformed_test_df = pd.DataFrame(transformed_test_data, columns=feature_names)
            transformed_test_df[target] = y_test.values
            
            # saving the preprocessor_obj
            save_obj(preprocessor_obj, self.preprocessor_config.preprocessor_path)
            
            # saving the transformed datasets to parquet format
            transformed_train_df.to_parquet(self.preprocessor_config.transformed_train_data_path, index=False)
            transformed_test_df.to_parquet(self.preprocessor_config.transformed_test_data_path, index=False)
            
            return(
                transformed_train_df,
                transformed_test_df,
                self.preprocessor_config.preprocessor_path
            )

            
        except Exception:
            logger.exception("Error transforming data from paths %s, %s", train_path, test_path)  
            raise  