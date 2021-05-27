import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import requests
import pickle

# Load the model
model = tensorflow.keras.models.load_model('models/pos/model.h5', compile=False)

with open("models/model.pkl", "rb") as f:
    model_value = pickle.load(f)

with open("models/scaler_x.pkl", "rb") as f:
    scaler_x = pickle.load(f)

with open("models/scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

labels = ["Внутриквартальное расположение", "На красной линии", "На первой линии второстепенной дороги"]

def value_property(data):
    price = {}
    error = 0.1

    data[13] = 0

    d = np.array(data)
    data_scaled = scaler_x.transform(d.reshape(1, -1))
    result_sell = model_value.predict(data_scaled)
    result_sell = scaler_y.inverse_transform(result_sell.reshape(1, -1))[0][0]

    price["sell"] = {}
    price["sell"]["avg"] = round(result_sell,2)
    price["sell"]["lower"] = round(result_sell - result_sell*error,2)
    price["sell"]["upper"] = round(result_sell + result_sell*error,2)

    data[13] = 1

    d = np.array(data)
    data_scaled = scaler_x.transform(d.reshape(1, -1))
    result_rent = model_value.predict(data_scaled)
    result_rent = scaler_y.inverse_transform(result_rent.reshape(1, -1))[0][0]

    price["rent"] = {}
    price["rent"]["avg"] = round(result_sell,2)
    price["rent"]["lower"] = round(result_sell - result_sell*error,2)
    price["rent"]["upper"] = round(result_sell + result_sell*error,2)

    return price


def get_position(url):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)[0].tolist()

    max_value = max(prediction)
    max_index = prediction.index(max_value)
    
    return {"label": max_index, "confidence": max_value}
