from fastapi import FastAPI
import pandas as pd
import numpy as np
from pydantic import BaseModel,Field,computed_field,field_validator
from typing import Literal ,Annotated
import pickle
from pathlib import Path
from fastapi.responses import JSONResponse


app = FastAPI()

# Load Artifact
BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "model" / "bmw_model.pkl", "rb") as file:
    artifact = pickle.load(file)

model = artifact["model"]
columns = artifact["columns"]
encoder = artifact["encoder"]
scaler = artifact["scaler"]


@app.get("/")
def home():
    return {"message":"BMW car price prediction API"}

@app.get("/health") 
def health_check():
    return{"status":"OK"}


class InputData(BaseModel):
    model: Annotated[Literal[' 5 Series', ' 6 Series', ' 1 Series', ' 7 Series',' 2 Series', ' 4 Series', ' X3', ' 3 Series',' X5', ' X4', ' i3', ' X1', ' M4',' X2', ' X6', ' 8 Series', ' Z4',' X7', ' M5', ' i8', ' M2',' M3', ' M6', ' Z3'],Field(...,description= "car model")] 

    transmission: Annotated[Literal["Manual", "Automatic", "Semi-Auto"],Field(...,description="transmissoin of car")]

    year:Annotated[int,Field(...,ge = 1990 , le = 2024)]
    mileage:Annotated[int,Field(...,ge=1,le=214000)]
    mpg:Annotated[float,Field(...,ge= 5.5,le = 470.8)]
    engineSize:Annotated[float,Field(...,ge=0.0,le=6.6)]
    tax:Annotated[int,Field(...,ge=0,le=580)]
    fuelType:Annotated[Literal["Petrol", "Diesel", "Hybrid","Other"],Field(...)]


    @field_validator("model")
    @classmethod
    def normalize_model(cls,value:str)-> str:
        value = str(value).capitalize()
        return value

    @field_validator("transmission")
    @classmethod
    def normalize_model(cls,value:str)-> str:
        value = str(value).capitalize()
        return value
    #feature engineering
    @computed_field
    @property
    def age(self)-> int:
        age = 2025 - self.year
        return age
    

@app.post("/predict")
def predict_premium(data:InputData):
    input_df = pd.DataFrame([{
                "fuelType": data.fuelType,
                "age": data.age,
                "model": data.model,
                "transmission": data.transmission,
                "mileage": data.mileage,
                "tax": data.tax,
                "engineSize": data.engineSize,
                "mpg": data.mpg
            }])
    cat_cols = ["transmission", "fuelType", "model"]
    num_cols = ["age", "mileage", "mpg", "engineSize", "tax"]
    # =========================
        # Encode Categorical
        # =========================

    encoded = encoder.transform(input_df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols)
    )

        # =========================
        # Scale Numerical
        # =========================

    scaled = scaler.transform(input_df[num_cols])

    scaled_df = pd.DataFrame(
        scaled,
        columns=num_cols
    )

        # =========================
        # Final Input
        # =========================

    final_df = pd.concat(
        [scaled_df, encoded_df],
        axis=1
    )

    # Match training columns exactly
    final_df = final_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(final_df)[0]
    price = np.expm1(prediction)
    return JSONResponse(status_code=200,content={"prediction":price})
