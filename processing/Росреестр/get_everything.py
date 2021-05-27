import pandas as pd
import requests
import time
import random
import re

# def get_object(ra, addr, sq):

#     if "г," in ra:
#         addr = ra + addr
#     else:
#         addr = "г. Москва, " + addr

        

#     r = requests.post("https://apiegrn.ru/api/cadaster/search", headers={"Token":"PMRT-OYJE-DL34-ZL2J"}, data={
#         "query": addr,
#         "mode": "normal"
#     }).json()

#     if r["objects"] == None:
#         return -1

#     diffs = []

#     for i in r["objects"]:
#         if i["AREA"] != "":
#             if i["AREA"] == ' кв.м':
#                 diffs.append(100000000)
#             else:
#                 diffs.append(abs(float( i["AREA"].replace(" кв.м", "").replace(",", ".")) - sq))
#         else:
#             diffs.append(100000000)
    
#     m = diffs.index(min(diffs))
#     return r["objects"][m]["CADNOMER"]

# def get_object_data(cad):
#     r = requests.post("https://apiegrn.ru/api/cadaster/objectInfoFull", headers={"Token":"PMRT-OYJE-DL34-ZL2J"}, data={
#         "query": cad,
#         "deep": 1
#     })
#     return r

    



# d = pd.read_csv("final8.csv", sep=";")
# di = d[(d["кадар"].isna())].drop_duplicates(subset=['Addr'])
# print(len(di))

# for i in di.index:
#     d.loc[d['Addr'] == di['Addr'][i], 'кадар'] = get_object(di["Метро/Район/Город"][i], di["Addr"][i], di["Общая пл."][i])
#     if i % 10 == 0:
#         d.to_csv("final8.csv", index=False, sep=";")

def get_metro(o_sh, o_d):
    r = requests.get("https://geocode-maps.yandex.ru/1.x/?geocode={1},{0}&kind=metro&format=json&result=1&apikey=66b4b9ee-2709-4c34-86a9-025d442f5a84".format(o_d, o_sh)).json()
    if "statusCode" not in r:
        for i in range(len(r["response"]["GeoObjectCollection"]["featureMember"])):
            if "метро" in r["response"]["GeoObjectCollection"]["featureMember"][i]["GeoObject"]["name"]:
                coords = r["response"]["GeoObjectCollection"]["featureMember"][i]["GeoObject"]["Point"]["pos"].split(" ")
                return float(coords[1]), float(coords[0])
        return 0, 0
    else:
        exit()
    

def get_time(m_sh, m_d, o_sh, o_d):
    r = requests.get("https://api.openrouteservice.org/v2/directions/foot-walking?api_key=5b3ce3597851110001cf624872823e3ca5c54cc58910b97bb6174265&start={1},{0}&end={3},{2}".format(m_d, m_sh, o_d, o_sh)).json()
    #print("https://api.openrouteservice.org/v2/directions/foot-walking?api_key=5b3ce3597851110001cf624872823e3ca5c54cc58910b97bb6174265&start={1},{0}&end={3},{2}".format(m_d, m_sh, o_d, o_sh))
    if "error" in r:
        return -1
    else:
        return r["features"][0]["properties"]["summary"]["duration"]

d = pd.read_csv("final8.csv", sep=";")
di = d[(d["До_метро"] == -1.0)].drop_duplicates(subset=['Addr'])
print(len(di))

j = 0
for i in di.index:
    j = j + 1
    print(j)
    cords = input("{0} :".format(di['Addr'][i]))
    cs = cords.split(', ')
    d.loc[d['Addr'] == di['Addr'][i], 'Широта'] = float(cs[0])
    d.loc[d['Addr'] == di['Addr'][i], 'Долгота'] = float(cs[1])
    metro_sh, metro_d = get_metro(float(cs[1]), float(cs[0]))

    ti = round(get_time(metro_d, metro_sh, float(cs[1]), float(cs[0])), 3)
    
    if ti == -1:
        ti = float((input("Сколько минут: ")))*60

    d.loc[d['Addr'] == di['Addr'][i], 'До_метро'] = ti
    d.to_csv("final8.csv", index=False, sep=";")