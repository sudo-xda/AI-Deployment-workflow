import requests

url = "http://127.0.0.1:5008/predict_api"

data = {
    "citric_acid": 0.22,
    "residual_sugar": 2.7,
    "pH": 3.28,
    "sulphates": 0.98,
    "alcohol": 9.9
}

#data=[[0.22,2.70,3.28,0.98,9.9]]
response = requests.post(url, json=data)

print("Wine Classified as class:", response.json())