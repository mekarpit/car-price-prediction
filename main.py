from pydantic import BaseModel
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

model_file = open("model.pickle","rb")
model_rf = pickle.load(model_file)

mean_data = pd.read_json('mean.json')
var_data = pd.read_json('var.json')
col_data = pd.read_json('columns.json')

class PredictionItem(BaseModel):
    age: int
    km_driven: int
    mileage: float
    engine: int                             
    max_power: int
    seats: int
    c_brand: str
    c_full_name: str
    c_seller_type: str
    c_fuel_type: str
    c_transmission_type: str


def predict_price(age, km_driven, mileage, engine, max_power, seats, c_brand, c_full_name, c_seller_type, c_fuel_type, c_transmission_type):    
    
    z = np.zeros(len(col_data))
    z[0] = (age-mean_data[0][0])/np.sqrt(var_data[0][0])
    z[1] = (km_driven-mean_data[0][1])/np.sqrt(var_data[0][1])
    z[2] = (mileage-mean_data[0][2])/np.sqrt(var_data[0][2])
    z[3] = (engine-mean_data[0][3])/np.sqrt(var_data[0][3])
    z[4] = (max_power-mean_data[0][4])/np.sqrt(var_data[0][4])
    z[5] = (seats-mean_data[0][5])/np.sqrt(var_data[0][5])


    brand = ("brand_"+c_brand)
    i_brand = col_data[col_data['data_columns']==brand].index.values[0]       
    z[i_brand] = 1

    fullname =("full_name_"+c_full_name)
    i_fullname = col_data[col_data['data_columns']==fullname].index.values[0] 
    z[i_fullname] = 1

    seller_type =("seller_type_"+c_seller_type)
    i_seller_type = col_data[col_data['data_columns']==seller_type].index.values[0]       
    z[i_seller_type] = 1

    fuel_type =("fuel_type_"+c_fuel_type)
    i_fuel_type = col_data[col_data['data_columns']==fuel_type].index.values[0]       
    z[i_fuel_type] = 1

    transmission_type =("transmission_type_"+c_transmission_type)
    i_transmission_type = col_data[col_data['data_columns']==transmission_type].index.values[0]        
    z[i_transmission_type] = 1
   
    return model_rf.predict([z])[0]


@app.post('/test')
def pred_endpoint(postdata:PredictionItem):
    inp_val = list(postdata.dict().values())
    print(inp_val)
    return predict_price(*inp_val)

@app.get("/")
async def root():
    return {"message": "Welcome to KAS Car Price Prediction"}