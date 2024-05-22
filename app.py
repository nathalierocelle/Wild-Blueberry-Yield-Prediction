from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import json
import numpy as np

model = joblib.load("final_model.joblib")

app = FastAPI()

class Data(BaseModel):
    clonesize : float
    honeybee : float 
    bumbles: float
    andrena: float
    osmia : float
    AverageOfLowerTRange: float
    AverageRainingDays: float
    fruitset: float 
    fruitmass: float
    seeds: float 
    
    def convert_json_to_array(self):
        return np.array([[self.clonesize, self.honeybee, self.bumbles, self.andrena, self.osmia, self.AverageOfLowerTRange,
                          self.AverageRainingDays, self.fruitset, self.fruitmass, self.seeds]])

@app.get("/")
def home():
    return "Wildberry Prediction API"

@app.post("/predict")
def predict_model(data:Data):
    array = data.convert_json_to_array()
    prediction = round(model.predict(array)[0],2)
    return {"status": "success", "data": data, "prediction": prediction}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=2, reload=True)