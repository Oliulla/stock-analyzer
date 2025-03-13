from fastapi import FastAPI
from pydantic import BaseModel
from predictor import get_stock_prediction

# Initialize FastAPI app
app = FastAPI()

# Create a model to receive the stock symbol as input
class StockPredictionRequest(BaseModel):
    stock_symbol: str

# FastAPI route to get the prediction
@app.post("/predict/")
async def predict_stock(data: StockPredictionRequest):
    stock_symbol = data.stock_symbol
    
    # Call the prediction function from predictor.py
    prediction = get_stock_prediction(stock_symbol)
    
    # Return the result
    return prediction

# To run the FastAPI app, use the following command in terminal:
# uvicorn api:app --reload
