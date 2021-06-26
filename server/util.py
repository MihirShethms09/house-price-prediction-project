import pickle
import json
import numpy as np


data_columns = None
locations = None
model = None

def get_estimated_price(location,total_sqft,bhk,bath,balcony):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bhk
    x[2] = bath
    x[3] = balcony
    if(loc_index>=4):
        x[loc_index] = 1
    
    return round(model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global data_columns
    global locations

    with open('./artifacts/columns.json','r') as f:
        data_columns = json.load(f)['data_columns']
        locations = data_columns[4:]

    global model
    if model is None:
        with open('./artifacts/house_price_prediction_model.pickle','rb') as f:
            model = pickle.load(f)

    print("loading saved artifacts...done")

def get_column_names():
    return data_columns

def get_location_names():
    return locations

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3,2))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2,1))
    print(get_estimated_price('Kalhalli', 1000, 2, 2,1)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2,1))  # other location