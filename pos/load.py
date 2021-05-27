import urllib.request
import pandas as pd
import os
import time
import random


d = pd.read_csv("final3.csv")
d = d.drop(d[d["pos"] == "-"].index)

for i,row in d.iterrows():
    if i in [103]:
        time.sleep(random.randint(0, 2))
        URL = "http://static-maps.yandex.ru/1.x/?ll={0},{1}&pt={0},{1}&spn=0.0027,0.0027&l=map&lang=en_US".format(row["Долгота"], row["Широта"])
        urllib.request.urlretrieve(URL, str(i)+".png")