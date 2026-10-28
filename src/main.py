import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.inference import InferencePipeline
from typing import Annotated
from src.logging_config import logger
from src.utils import load_obj
from src.schemas import BankruptsyInput, DataOut
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "model.pkl"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the model once when the app starts
    logger.info("Loading bankruptcy prediction model")
    app.state.inference_pipeline= InferencePipeline(model_path=MODEL_PATH)
    
    logger.info("Model loaded successfully")
    yield
    logger.info("Application shutting down")

# To inject the pipeline     
def get_inference_pipeline(request:Request):
    return request.app.state.inference_pipeline
        

app= FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "welcome to bankruptsy in ploand api"}

@app.post("/predict", response_model=DataOut)
def predict(input_data: BankruptsyInput, inference_pipeline:Annotated[InferencePipeline, Depends(get_inference_pipeline)]):
    try:

        # inference_pipeline = InferencePipeline(model_path="artifacts/model.pkl")
        prediction = inference_pipeline.predict(input_data=input_data)
        return DataOut(
            success=True,
            prediction=int(prediction),
            message="Prediction completed successfully!"
        )

    except Exception as e:
        logger.exception("Bankruptcy prediction failed")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during prediction",
        ) from e
           

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)    