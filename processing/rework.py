import pandas as pd
import requests

data = pd.read_csv("final8.csv", sep=";")

#Ключи 66b4b9ee-2709-4c34-86a9-025d442f5a84 fbf80273-9977-487f-8c83-85710418e61f b708aee3-e718-497c-a44b-9e908c929544

for i in range(27202, len(data)):
    if "г," in data["Метро/Район/Город"][i]:
        adr = data["Метро/Район/Город"][i] + " " + data["Addr"][i]
    else:
        adr = data["Addr"][i]

    r = requests.get("https://geocode-maps.yandex.ru/1.x?geocode={0}&apikey=b708aee3-e718-497c-a44b-9e908c929544&format=json".format(adr)).json()

    if len(r['response']['GeoObjectCollection']['featureMember']) != 0:
        coords = r['response']['GeoObjectCollection']['featureMember'][0]["GeoObject"]["Point"]["pos"].split(" ")

        data.at[i, 'Долгота'] = float(coords[0])
        data.at[i, 'Широта'] = float(coords[1])
    else:
        data.at[i, 'Долгота'] = 0
        data.at[i, 'Широта'] = 0

    data.to_csv("d1.csv", sep=";", index=False)
