import requests
import json
import time
import random
from sklearn import metrics
import numpy as np

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    #mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    #print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

def get_coords(region, addr):
    if "г," in region:
        addr = region + " " + addr
    time.sleep(random.uniform(0.2, 0.5))
    j = requests.get("http://search.maps.sputnik.ru/search/addr?q={0}".format(addr)).content
    resp = json.loads(j)
    if "address" in resp["result"]:
        return resp["result"]["address"][0]["features"][0]["geometry"]["geometries"][0]["coordinates"][1], resp["result"]["address"][0]["features"][0]["geometry"]["geometries"][0]["coordinates"][0]
    else:
        return [0, 0]

def get_closest_metro(lng, lat):
    try:
        j = requests.get("https://json.smappi.org/milash/metro-station/getClosest?city=Москва&lat={0}&lon={1}&count=1".format(lat, lng)).content
        resp = json.loads(j)
        if "status" in resp:
            return None
        time.sleep(randint(1, 2))
        j = requests.get("http://footroutes.maps.sputnik.ru/?loc={0},{1}&loc={2},{3}".format(lat, lng, resp[0]["station"]["geo_lat"], resp[0]["station"]["geo_lon"])).content
        resp = json.loads(j)
        return resp["route_summary"]["total_time"]
    except Exception:
        return -1
    
    

