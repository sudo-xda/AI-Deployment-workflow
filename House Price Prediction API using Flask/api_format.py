import requests
#url = 'http://127.0.0.1:5000/predict_api' 
url='https://4695-103-106-200-60.ngrok-free.app/predict_api'
data = {'year': 2025, 'sqft': 1}  
response = requests.post(url, json=data)  
try:
    print(response.json())  
except ValueError:
    print("Error: Response is not JSON. Got:", response.text)