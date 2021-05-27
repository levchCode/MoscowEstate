import ml
import requests
from datetime import datetime

def get_coords(addr):
    resp = requests.get("http://search.maps.sputnik.ru/search/addr?q={0}".format(addr)).json()

    if "address" in resp["result"]:
        return resp["result"]["address"][0]["features"][0]["geometry"]["geometries"][0]["coordinates"][1], resp["result"]["address"][0]["features"][0]["geometry"]["geometries"][0]["coordinates"][0]
    else:
        return 0, 0


def get_metro(lat, lon):
    try:
        resp = requests.get("https://json.smappi.org/milash/metro-station/getClosest?city=Москва&lat={0}&lon={1}&count=1".format(lat, lon)).json()

        if "status" in resp:
            return None

        j = requests.get("http://footroutes.maps.sputnik.ru/?loc={0},{1}&loc={2},{3}".format(lat, lon, resp[0]["station"]["geo_lat"], resp[0]["station"]["geo_lon"])).json()
        return resp["route_summary"]["total_time"]
    except Exception:
        return -1

def get_position(d, sh):
    URL = "http://static-maps.yandex.ru/1.x/?ll={0},{1}&pt={0},{1}&spn=0.0027,0.0027&l=map".format(float(sh), float(d))
    pos = ml.get_position(URL)
    return pos, URL

def get_intermed_data(data):
    resp = {}
    resp["cords"] = {}
    resp["cords"]["sh"], resp["cords"]["d"] = get_coords(data["addr"])
    print(resp)
    resp["metro"] = get_metro(resp["cords"]["sh"], resp["cords"]["d"])
    resp["pos"], resp["pos"]["img"] = get_position(resp["cords"]["sh"], resp["cords"]["d"])
    return resp, data

def value_property(data):
    print(data)
    d = []

    date = datetime.strptime(data["date"], "%Y-%m-%d")

    d.append(date.day)
    d.append(date.month)
    d.append(date.year)
    d.append(int(data["purp"]))
    d.append(int(data["type"]))
    d.append(int(data["agreement"]))
    d.append(int(data["class"]))
    d.append(float(data["sq"]))
    d.append(int(data["floor"]))
    d.append(int(data["storyes"]))
    d.append(int(data["seller"]))
    d.append(float(data["d"]))
    d.append(float(data["sh"]))
    d.append(int(data["op"]))
    d.append(int(data["metro"]))
    d.append(data["pos"])

    info = ml.value_property(d)
    return info