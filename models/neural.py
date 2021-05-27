from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.utils import validation
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.core import Dropout

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('R^2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', mse)
    print('RMSE: ', round(np.sqrt(mse),4))

data = pd.read_csv('final10.csv', sep=";", parse_dates=["Дата"])
data = data[["Дата", "Addr", "Назначение", "Тип объекта", "Класс помещ.", "Общая пл.", "Этаж", "Этажность", "Широта", "Долгота", "До_метро", "Операция", "Комментарий", "Цена", "pos", "Ближ_цена", "Район"]]

data["День"] = pd.to_datetime(data["Дата"]).dt.day
data["Месяц"] = pd.to_datetime(data["Дата"]).dt.month
data["Год"] = pd.to_datetime(data["Дата"]).dt.year
data["Район"] = data["Район"].astype("str")


data['Дата'] = (data["Дата"] - pd.to_datetime('1970-01-01')).dt.total_seconds()

data = data[data["Назначение"] == "Офис"]

data = data[data["Долгота"] != 0]
data = data[data["Широта"] != 0]


data['parking'] = np.where(data['Комментарий'].str.contains('|'.join(["парковк", "паркин", "стоян"])), 1, 0)
data['security'] = np.where(data['Комментарий'].str.contains('|'.join(["охранa", "охраняемая тер"])), 1, 0)
#data['repair'] = np.where(data['Комментарий'].str.contains('|'.join(["ремон"])), 1, 0)
data['internet'] = np.where(data['Комментарий'].str.contains('|'.join(["интернет"])), 1, 0)
data['phone'] = np.where(data['Комментарий'].str.contains('|'.join(["телефо"])), 1, 0)
data['access'] = np.where(data['Комментарий'].str.contains('|'.join(["пропуск"])), 1, 0)
data['furniture'] = np.where(data['Комментарий'].str.contains('|'.join(["мебелью", "есть мебе", "имеется мебе"])), 1, 0)
data['canteen'] = np.where(data['Комментарий'].str.contains('|'.join(["столовая"])), 1, 0)

data = data.reset_index()


le = preprocessing.LabelEncoder()
le.fit(data["Назначение"])
data["Назначение"] = le.transform(data["Назначение"])

le = preprocessing.LabelEncoder()
le.fit(data["pos"])
data["pos"] = le.transform(data["pos"])


le = preprocessing.LabelEncoder()
le.fit(data["Тип объекта"])
data["Тип объекта"] = le.transform(data["Тип объекта"])

le = preprocessing.LabelEncoder()
le.fit(data["Класс помещ."])
data["Класс помещ."] = le.transform(data["Класс помещ."])

le = preprocessing.LabelEncoder()
le.fit(data["Операция"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
data["Операция"] = le.transform(data["Операция"])

le = preprocessing.LabelEncoder()
le.fit(data["Этаж"])
data["Этаж"] = le.transform(data["Этаж"])

le = preprocessing.LabelEncoder()
le.fit(data["Район"])
data["Район"] = le.transform(data["Район"])

data = data[data["Операция"] == 0]

data["Цена"] = np.where(data["Цена"] < data['Цена'].quantile(0.10), data['Цена'].quantile(0.10), data['Цена'])
data['Цена'] = np.where(data["Цена"] > data['Цена'].quantile(0.90), data['Цена'].quantile(0.90), data['Цена'])

data["До_метро"] = np.where(data["До_метро"] < data['До_метро'].quantile(0.10), data['До_метро'].quantile(0.10), data['До_метро'])
data['До_метро'] = np.where(data["До_метро"] > data['До_метро'].quantile(0.90), data['До_метро'].quantile(0.90), data['До_метро'])

data["Этажность"] = np.where(data["Этажность"] < data['Этажность'].quantile(0.10), data['Этажность'].quantile(0.10), data['Этажность'])
data['Этажность'] = np.where(data["Этажность"] > data['Этажность'].quantile(0.90), data['Этажность'].quantile(0.90), data['Этажность'])

data["Долгота"] = np.where(data["Долгота"] < data['Долгота'].quantile(0.10), data['Долгота'].quantile(0.10), data['Долгота'])
data['Долгота'] = np.where(data["Долгота"] > data['Долгота'].quantile(0.90), data['Долгота'].quantile(0.90), data['Долгота'])

data["Широта"] = np.where(data["Широта"] < data['Широта'].quantile(0.10), data['Широта'].quantile(0.10), data['Широта'])
data['Широта'] = np.where(data["Широта"] > data['Широта'].quantile(0.90), data['Широта'].quantile(0.90), data['Широта'])

rent = data[["Цена", "Addr", "Дата", "Тип объекта", "Класс помещ.", "Этаж", "Этажность", "Широта", "Долгота", "До_метро", "pos", "Операция", "parking", "security", "internet", "phone", "access", "furniture", "canteen", "Ближ_цена", "Район"]]


x = rent[["Дата", "Тип объекта", "Класс помещ.", "Этаж", "Этажность", "Широта", "Долгота", "До_метро", "pos", "parking", "security", "internet", "phone", "access", "furniture", "canteen", "Район"]]
Y = rent[["Цена"]]


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
y_scaled = min_max_scaler.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled.ravel(), test_size=0.15)

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

model = keras.Sequential([
      layers.BatchNormalization(),
      layers.Dense(100, kernel_initializer='normal', activation='gelu'),
      layers.Dense(100, kernel_initializer='normal', activation='relu'),
      layers.Dense(1)
  ])

model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=64)

pred = model.predict(X_test)
regression_results(y_test, pred)

