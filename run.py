import requests

API_KEY = "YOUR_API_KEY"

url = "https://stock.indianapi.in/stock"

headers = {
    "X-Api-Key": 'sk-live-e569mSP3TtIpVxn5gxUeOYgOnFVzKKiVAGKtBreH'
}

params = {
    "name": "Reliance"
}

response = requests.get(url, headers=headers, params=params)

print(response.status_code)
print(response.json())