# client 

import requests

url = 'http://localhost:9696/score'

data = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

resp = requests.post(url, json=data, timeout=10)
print(resp.status_code, resp.text)
