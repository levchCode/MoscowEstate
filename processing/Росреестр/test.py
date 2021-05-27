
from textwrap import indent
import pandas as pd
import requests
import time
import random
from dadata import Dadata




# # import tensorflow.keras
# # from PIL import Image, ImageOps
# # import numpy as np

# # # # Load the model
# # # model = tensorflow.keras.models.load_model('pos/model.h5', compile=False)

# # # def get_position(url):
# # #     # Create the array of the right shape to feed into the keras model
# # #     # The 'length' or number of images you can put into the array is
# # #     # determined by the first position in the shape tuple, in this case 1.
# # #     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# # #     # Replace this with the path to your image
# # #     image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

# # #     #resize the image to a 224x224 with the same strategy as in TM2:
# # #     #resizing the image to be at least 224x224 and then cropping from the center
# # #     size = (224, 224)
# # #     image = ImageOps.fit(image, size, Image.ANTIALIAS)

# # #     #turn the image into a numpy array
# # #     image_array = np.asarray(image)

# # #     # Normalize the image
# # #     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# # #     # Load the image into the array
# # #     data[0] = normalized_image_array

# # #     # run the inference
# # #     prediction = model.predict(data)[0].tolist()

# # #     labels = ["внут", "красная", "плвд"]

# # #     max_value = max(prediction)
# # #     max_index = prediction.index(max_value)
    
# # #     return labels[max_index]

# # # d = pd.read_csv("d3.csv", sep=";")


# # # for i in d.index:
# # #     if pd.isna(d["pos"][i]):
# # #         print(i)
# # #         time.sleep(random.uniform(0.1,0.5))
# # #         URL = "http://static-maps.yandex.ru/1.x/?ll={0},{1}&pt={0},{1}&spn=0.0027,0.0027&l=map".format(d["Долгота"][i], d["Широта"][i])
# # #         d.loc[(d['Широта'] == d["Широта"][i])&(d['Широта'] == d["Широта"][i]), "pos"] = get_position(URL)


# # # d.to_csv("d3.csv", index=False, sep=";")


# #key = "PMRT-OYJE-DL34-ZL2J"
# #Ключи 66b4b9ee-2709-4c34-86a9-025d442f5a84 fbf80273-9977-487f-8c83-85710418e61f b708aee3-e718-497c-a44b-9e908c929544

# key = '39e3df37-7b6f-4a44-8a64-d3471f4ffd3d'

def get_reg(o_sh, o_d):

    r = requests.get("https://geocode-maps.yandex.ru/1.x/?geocode={1},{0}&kind=district&format=json&result=1&apikey=66b4b9ee-2709-4c34-86a9-025d442f5a84".format(o_d, o_sh)).json()
    if "statusCode" not in r:
        if len(r["response"]["GeoObjectCollection"]["featureMember"]) > 0:
            return r["response"]["GeoObjectCollection"]["featureMember"][len(r["response"]["GeoObjectCollection"]["featureMember"])-1]["GeoObject"]["name"]
        else:
            return 0
    else:
        exit()
    

# def get_time(m_sh, m_d, o_sh, o_d):
#     time.sleep(random.uniform(0.5,1.2))
#     r = requests.get("https://api.openrouteservice.org/v2/directions/foot-walking?api_key=5b3ce3597851110001cf624872823e3ca5c54cc58910b97bb6174265&start={1},{0}&end={3},{2}".format(m_d, m_sh, o_d, o_sh)).json()
#     #print("https://api.openrouteservice.org/v2/directions/foot-walking?api_key=5b3ce3597851110001cf624872823e3ca5c54cc58910b97bb6174265&start={1},{0}&end={3},{2}".format(m_d, m_sh, o_d, o_sh))
#     if "error" in r:
#         return -1
#     else:
#         return r["features"][0]["properties"]["summary"]["duration"]



d = pd.read_csv("final9.csv", sep=";")
di = d.drop_duplicates(subset=['Addr'])


j = 0

print(d[d["Район"] == "0"])
for i in range(2099, len(di.index)):


    if d["Район"][i] == 0:
        reg = get_reg(di["Долгота"][i], di["Широта"][i])
        d.loc[d['Addr'] == di['Addr'][i], 'Район'] = reg

    j = j + 1

    if j % 100 == 0:
        print(i)
        d.to_csv("final9.csv", index=False, sep=";")

        
    #     if type(t) == float:
    #         d.at[i, 'До_метро'] = -1
    #     elif "П" in t:
    #         d.at[i, 'До_метро'] = int(t.replace("П", ""))*60
    #     elif "Т" in t:
    #         d.at[i, 'До_метро'] = int(t.replace("Т", ""))*60
    #     elif "км" in t:
    #          d.at[i, 'До_метро'] = (int(t.replace("км", ""))/5)*60*60
    # else:
    #     d.at[i, 'До_метро'] = round(get_time(metro_d, metro_sh, d["Долгота"][i], d["Широта"][i]), 3)

    if i % 100 == 0:
        print(i)
    
d.to_csv("d3.csv", index=False, sep=";")



