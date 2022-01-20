# Import the necessary modules from FastAPI
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# Import the PyCaret Regression module
import pycaret.classification as pycr

# Import other necessary packages
import pandas as pd
import os

class Features(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: int
    ST_Slope: str
    HeartDisease: Optional[str] = None

# Create a class to store the deployed model & use it for prediction
class Model:
    def __init__(self, modelname):
        """
        To initalize the model
        modelname: Name of the model
        """
        # Load the deployed model
        self.model = pycr.load_model(os.path.join("..", "models", modelname))
    
    def predict(self, data):
        """
        To use the loaded model to make predictions on the data
        data: Pandas DataFrame to perform predictions
        """
        # Return the column containing the predictions  
        # (i.e. 'Label') after converting it to a list
        predictions = pycr.predict_model(self.model, data=data).Label.to_list()
        return predictions

# Load the model
model = Model("model")

# Initialize the FastAPI application
app = FastAPI()

# Create the POST endpoint for individual prediction
@app.post("/predict/individual")
async def predict_individual(features: Features):
    # Use the features to create a dataframe to pass the model as input
    data = pd.DataFrame.from_records([features.dict()])
    # Return a JSON object containing the model predictions
    return {
        "Labels": model.predict(data)
    } 


# Create the POST endpoint for batch prediction
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    print(file.filename)
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):
        # Create a temporary file with the same name as the uploaded 
        # CSV file to load the data into a pandas Dataframe
        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        print(data.shape)
        os.remove(file.filename)
        # Return a JSON object containing the model predictions
        return {
            "Labels": model.predict(data)
        }    
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request 
        # (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")
