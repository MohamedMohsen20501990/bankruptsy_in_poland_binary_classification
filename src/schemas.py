from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator
from src.utils import load_obj, generate_input_features

# Features schema to be validated by pydantic
FEATURES_X =[f"X{i}"for i in range(1,65)]
MODEL_FEATURES = FEATURES_X + ["Year"]

class BankruptsyInput(BaseModel):
    """Validate the financial features required for bankruptsy prediction"""
    
    # Prevent the user to enter any other fields which is not provided by the BaseModel[features]   
    model_config = ConfigDict(extra="forbid")
    
    # Features attribute
    features :dict[str, FiniteFloat]= Field(description="64 financial ratios + Year feature")
    
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
    
class DataOut(BaseModel):
    success:bool
    prediction: int | None = None
    message: Annotated[str, Field(min_length=1, max_length=200)]
    
        