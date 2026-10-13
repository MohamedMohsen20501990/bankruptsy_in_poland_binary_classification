import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import math
import pandas as pd
from src.logging_config import logger
from src.utils import load_obj, generate_input_features

from pydantic import BaseModel, ConfigDict, model_validator, Field

# Feature schema
features_x = [f"X{i}" for i in range(1, 65)]
model_features = features_x + ["Year"]

class BankruptsyInput(BaseModel):
    model_config=ConfigDict(extra="forbid")
    features: dict[str, float] = Field(description="Financial ratios and year")
    
    @model_validator(mode="after")
    def validate_features(self):

        expected = set(model_features)
        provided = set(self.features)
        missing = expected - provided
        extra = provided - expected
        
        if missing:
            raise ValueError(f"Missing required features {sorted(missing)}")
        if extra:
            raise ValueError(f"Unexpected features {sorted(extra)}")
        
        for name, value in self.features.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        return self    
            
class InputData:
    def __init__(self, data:BankruptsyInput):
        self.data=data
        
    def to_data_frame(self):
        df = pd.DataFrame([[self.data.features[col] for col in model_features]], columns=model_features)
        return df

class InferencePipeline:
    def __init__(self, model_path:Path, preprocessor_path: Path):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = load_obj(self.model_path)
        self.preprocessor = load_obj(self.preprocessor_path)
        
    def predict(self, data: BankruptsyInput):
        input_data = InputData(data)
        df = input_data.to_data_frame()
        
        transformed_data = self.preprocessor.transform(df)
        prediction = self.model.predict(transformed_data)
        return {"Prediction": int(prediction[0])}
    
  
  
  
if __name__ == "__main__":            
    user_input = generate_input_features(rown_index=0, year=2.0)

    validated_features = BankruptsyInput(features = user_input)
 
    inference_pipeline = InferencePipeline(model_path="artifacts/model.pkl", preprocessor_path="artifacts/preprocessor.pkl")
    result = inference_pipeline.predict(validated_features)
    print(result)

                  
