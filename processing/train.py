# Основной скрипт для первичной обработки и обучения моделей

from os import X_OK
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sn
import requests
import time

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

data = pd.read_csv('final11.csv', sep=";", parse_dates=["Дата"])
data2 = pd.read_csv('final9.csv', sep=";")
data["Ближ_аналог"] = data2["Ближ_ана"]
data = data[["Дата", "Addr", "Назначение", "Тип объекта", "Класс помещ.", "Общая пл.", "Этаж", "Этажность", "Широта", "Долгота", "До_метро", "Операция", "Комментарий", "Цена", "pos", "Ближ_цена", "Ближ_аналог", "Район"]]

df = data["Дата"]

# data["День"] = pd.to_datetime(data["Дата"]).dt.day
# data["Месяц"] = pd.to_datetime(data["Дата"]).dt.month
# data["Год"] = pd.to_datetime(data["Дата"]).dt.year
data["Район"] = data["Район"].astype("str")



#data['Дата'] = (data["Дата"] - pd.to_datetime('1970-01-01')).dt.total_seconds()

#data = data[data["Назначение"] == "Офис"]

data = data[data["Долгота"] != 0]
data = data[data["Широта"] != 0]

#data = data.drop_duplicates(["Addr"])

# text = data["Комментарий"].str.cat(sep=' ')
# f = open("all_comments.txt", "w", encoding="UTF-8")
# f.write(text)
# f.close()


# from wordcloud import WordCloud
# wordcloud = WordCloud(width=800, height=400).generate(text)
# import matplotlib.pyplot as plt
# import matplotlib

# plt.figure( figsize=(20,10), facecolor='k')
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.savefig("f.png")

data['parking'] = np.where(data['Комментарий'].str.contains('|'.join(["парковк", "паркин", "стоян"])), 1, 0)
data['security'] = np.where(data['Комментарий'].str.contains('|'.join(["охранa", "охраняемая тер"])), 1, 0)
#data['repair'] = np.where(data['Комментарий'].str.contains('|'.join(["ремон"])), 1, 0)
data['internet'] = np.where(data['Комментарий'].str.contains('|'.join(["интернет"]), case=False, regex=False), 1, 0)
data['phone'] = np.where(data['Комментарий'].str.contains('|'.join(["телефо"])), 1, 0)
data['access'] = np.where(data['Комментарий'].str.contains('|'.join(["пропуск"])), 1, 0)
data['furniture'] = np.where(data['Комментарий'].str.contains('|'.join(["мебелью", "есть мебе", "имеется мебе"])), 1, 0)
data['canteen'] = np.where(data['Комментарий'].str.contains('|'.join(["столовая"])), 1, 0)

data = data.reset_index()


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

# print(data.columns)

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

rent = data[["Цена", "Дата", "Тип объекта", "Класс помещ.", "Этаж", "Этажность", "Широта", "Долгота", "До_метро", "pos", "Операция", "parking", "security", "internet", "phone", "access", "furniture", "canteen", "Ближ_цена", "Ближ_аналог",  "Район"]]

x = rent[["Дата", "Тип объекта", "Класс помещ.", "Этаж", "Этажность", "Широта", "Долгота", "До_метро", "pos", "internet", "phone", "access", "furniture", "canteen", "Район", "Ближ_аналог"]]
Y = rent[["Цена"]]


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
y_scaled = min_max_scaler.fit_transform(Y)


X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled.ravel(), test_size=0.15)

# def mae(y_true, y_pred):
#     return np.mean(abs(y_true - y_pred))

# baseline_guess = np.median(y_train)

# print('The baseline guess is a score of %0.2f' % baseline_guess)
# print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))


# nsvr = SVR(kernel="rbf", C=2)
# nsvr.fit(X_train, y_train)
# y_pred = nsvr.predict(X_test)
# regression_results(y_test, y_pred)

# knn = KNeighborsRegressor(n_neighbors=5, weights='distance', p=1)
# # # knn = GridSearchCV(KNeighborsRegressor(), {
# # #     "n_neighbors":[i for i in range(3, 100, 5)],
# # #     "weights": ('uniform', 'distance'),
# # #     "p":[1,2],
# # # })
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# regression_results(y_test, y_pred)

# tre = DecisionTreeRegressor(criterion='mse', min_samples_split=5, max_depth=48).fit(X_train, y_train)
# y_pred = tre.predict(X_test)
# regression_results(y_test, y_pred)

# dtr = RandomForestRegressor(n_estimators=1000, criterion="mse").fit(X_train, y_train)
# y_pred = dtr.predict(X_test)
# # regression_results(y_test, y_pred)

# rfr = RandomForestRegressor(200, criterion="mse")

# # # # # #     "n_estimators":[100, 130],# 170, 210, 250, 320, 360, 430, 460, 500],
# # # # # #     "criterion": ("mse", "mae"),
    
# # # # # # # })
# rfr.fit(X_train, y_train)
# y_pred = rfr.predict(X_test)
# # regression_results(y_test, y_pred)

ab = CatBoostRegressor(iterations=3000, eval_metric="R2", loss_function="RMSE", learning_rate=0.07)
ab.fit(X_train, y_train, eval_set=(X_test, y_test))
y_pred = ab.predict(X_test)
regression_results(y_test, y_pred)



# # # ab = MLPRegressor()
# # # ab.fit(X_rent_train, y_rent_train)
# # # y_pred = ab.predict(X_rent_test)
# # # regression_results(y_rent_test, y_pred)

from scipy import stats

def get_prediction_interval(prediction, y_test, test_predictions, pi=.95):
    '''
    Get a prediction interval for a linear regression.
    
    INPUTS: 
        - Single prediction, 
        - y_test
        - All test set predictions,
        - Prediction interval threshold (default = .95) 
    OUTPUT: 
        - Prediction interval for single prediction
    '''
    
    #get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
#get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
    
    
#generate prediction interval lower and upper bound cs_24
    lower, upper = prediction - interval, prediction + interval
    return lower, prediction, upper


l, p, u = get_prediction_interval(y_pred[0], y_test, y_pred)


print(min_max_scaler.inverse_transform(np.array(l).reshape(-1, 1)))
print(min_max_scaler.inverse_transform(np.array(p).reshape(-1, 1)))
print(min_max_scaler.inverse_transform(np.array(u).reshape(-1, 1)))

print(min_max_scaler.inverse_transform(y_test[0].reshape(-1, 1)))