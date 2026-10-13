from pathlib import Path
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from src.utils import load_obj, generate_input_features

# Features schema to be validated by pydantic
FEATURES_X =[f"X{i}"for i in range(1,65)]
MODEL_FEATURES = FEATURES_X + ["Year"]

class BankruptsyInput(BaseModel):
    """Validate the financial features required for bankruptsy prediction"""\
    
    # Prevent the user to enter any other fields which is not provided by the BaseModel[features]   
    model_config = ConfigDict(extra="forbid")
    
    # Features attribute
    features :dict[str, FiniteFloat]= Field(description="64 financial rations and the year feature")
    
    @model_validator(mode="after")
    def validate_feature_names(self):
        expected = set(MODEL_FEATURES)
        provided = set(self.features)

        missing = expected - provided
        extra = provided - expected

        if missing or extra:
            errors = []

            if missing:
                errors.append(f"Missing features: {sorted(missing)}")

            if extra:
                errors.append(f"Unexpected features: {sorted(extra)}")

            raise ValueError("; ".join(errors))

        return self


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
        
        return {"prediction": int(prediction)}    
    
    
    
if __name__ == "__main__":
    
    user_input = generate_input_features(rown_index=0, year=3)
    
    validated_input = BankruptsyInput(features=user_input)
    
    inference_pipeline = InferencePipeline(model_path="artifacts/model.pkl")
    result = inference_pipeline.predict(input_data=validated_input)
    print(result)