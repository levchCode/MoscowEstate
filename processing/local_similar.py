import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv('final8.csv', sep=";", parse_dates=["Дата"])
data = data[["Дата", "Назначение", "Тип объекта", "Класс помещ.", "Общая пл.", "Этаж", "Этажность", "Широта", "Долгота", "До_метро", "Операция", "Цена", "pos", "Ближ_ана"]]

#Поиск объекта-аналога и средняя цена в радуусе км

cor = data
for i in range(len(data.index)):
    if pd.isna(data.iloc[i]["Ближ_ана"]):
        cur = data.iloc[i]

        d = data[((data["Широта"] != cur["Широта"]) & (data["Долгота"] != cur["Долгота"])) ]

        d = d[["Дата", "Назначение", "Тип объекта", "Класс помещ.", "Общая пл.", "Этаж", "Этажность", "Широта", "Долгота", "До_метро", "Операция", "Цена", "pos", "Месяц", "Год"]]
        cur = cur[["Дата", "Назначение", "Тип объекта", "Класс помещ.", "Общая пл.", "Этаж", "Этажность", "Широта", "Долгота", "До_метро", "Операция", "Цена", "pos", "Месяц", "Год"]]
        tree = BallTree(d, leaf_size=2)
        di, ind = tree.query([cur], k=1)
        found = data.iloc[ind[0][0]]


        time.sleep(0.1)

        try:
            r = requests.get("https://www.statbureau.org/calculate-inflation-price-json?country=russia&format=false&start={0}-{1}-01&end={2}-{3}-01&amount={4}&denominationsToApply[0]=1998-01-01".format(
            int(found["Год"]), int(found["Месяц"]), int(cur["Год"]), int(cur["Месяц"]),  found["Цена"]
            ))
        except Exception:
            r = requests.get("https://www.statbureau.org/calculate-inflation-price-json?country=russia&format=false&start={0}-{1}-01&end={2}-{3}-01&amount={4}&denominationsToApply[0]=1998-01-01".format(
            int(found["Год"]), int(found["Месяц"]), int(cur["Год"]), int(cur["Месяц"]),  found["Цена"]
            )) 


        data.loc[i, 'Ближ_ана'] = float(r.text.replace('"', ''))

        if i % 100 == 0:
            print(i)
            data["Дата"] = df
            data.to_csv("final8_sell.csv", sep=";", index=False)
            data['Дата'] = (data["Дата"] - pd.to_datetime('1970-01-01')).dt.total_seconds()


# from sklearn.neighbors import BallTree
# x = data
# data = data[data["Операция"] == "Продажа"]
# cor  = data[["Широта", "Долгота"]].drop_duplicates(["Широта", "Долгота"])
# tree_sell = BallTree(cor, metric="haversine", leaf_size=5)

# data = x

# data = data[data["Операция"] == "Сдаю"]
# cor  = data[["Широта", "Долгота"]].drop_duplicates(["Широта", "Долгота"])
# tree_rent = BallTree(cor, metric="haversine", leaf_size=5)

# data = x

# de_data = data.drop_duplicates(["Широта", "Долгота"])
# de_data = de_data.reset_index()
# print(len(de_data))



# for i in range(136, len(de_data.index)):
#     if de_data["Операция"][i] == "Сдаю":
#         ind = tree_rent.query_radius([de_data[["Широта", "Долгота"]].iloc[i]], r=0.003)
#     else:
#         ind = tree_sell.query_radius([de_data[["Широта", "Долгота"]].iloc[i]], r=0.003)
    
#     print(i, len(ind[0]))

#     s = 0
#     cur = data.iloc[i]
#     for j in ind[0]:
#         time.sleep(0.1)

#         found = data.iloc[j]
#         try:
#             r = requests.get("https://www.statbureau.org/calculate-inflation-price-json?country=russia&format=false&start={0}-{1}-01&end={2}-{3}-01&amount={4}&denominationsToApply[0]=1998-01-01".format(
#                     found["Год"], found["Месяц"], cur["Год"], cur["Месяц"],  found["Цена"]
#                 ))
#         except Exception:
#             r = requests.get("https://www.statbureau.org/calculate-inflation-price-json?country=russia&format=false&start={0}-{1}-01&end={2}-{3}-01&amount={4}&denominationsToApply[0]=1998-01-01".format(
#                     found["Год"], found["Месяц"], cur["Год"], cur["Месяц"],  found["Цена"]
#                 ))
#         s += float(r.text.replace('"', ''))
    
#     data.loc[found["Addr"] == data["Addr"], 'Ближ_цена'] = s/len(ind)

#     if (i + 1) % 100 == 0:
#         data["Дата"] = df
#         data.to_csv("final11.csv", sep=";", index=False)
#         data['Дата'] = (data["Дата"] - pd.to_datetime('1970-01-01')).dt.total_seconds()

    
# data.to_csv("final11.csv", sep=";", index=False)

data.to_csv("final8_sell.csv", sep=";", index=False)

