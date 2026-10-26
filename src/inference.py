from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.schemas import BankruptsyInput

from src.utils import load_obj, generate_input_features

# Features schema to be validated by pydantic
FEATURES_X =[f"X{i}"for i in range(1,65)]
MODEL_FEATURES = FEATURES_X + ["Year"]


class InferencePipeline:
    """load the fitted ML-pipeline and predict bankruptsy"""
    def __init__(self, model_path: str | Path):
        self.model = load_obj(model_path)
        
    @staticmethod
    def _prepare_input(data: BankruptsyInput) -> pd.DataFrame:
        """creates a dataframe with expected features order"""
        df =pd.DataFrame([data.features], columns=MODEL_FEATURES, dtype=float)
        
        return df
    
    def predict(self, input_data: BankruptsyInput) -> dict:
        df = self._prepare_input(input_data)
        prediction = self.model.predict(df)[0]
        
        return int(prediction)   
    
    
    
if __name__ == "__main__":
    
    user_input = generate_input_features(rown_index=0, year=3)
    
    validated_input = BankruptsyInput(features=user_input)
    
    inference_pipeline = InferencePipeline(model_path="artifacts/model.pkl")
    result = inference_pipeline.predict(input_data=validated_input)
    print(result)