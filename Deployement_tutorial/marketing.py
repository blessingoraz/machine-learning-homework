import requests

url = 'http://localhost:9696/predict'

customer = {
  "gender": "female",
  "seniorcitizen": 0,
  "partner": "yes",
  "dependents": "no",
  "phoneservice": "yes",
  "multiplelines": "no",
  "internetservice": "dsl",
  "onlinesecurity": "yes",
  "onlinebackup": "no",
  "deviceprotection": "yes",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "one_year",
  "paperlessbilling": "no",
  "paymentmethod": "mailed_check",
  "tenure": 10,
  "monthlycharges": 56.95,
  "totalcharges": 20000
}

response = requests.post(url, json=customer)
churn = response.json()

print('prob of churning is:', churn)

if churn['churn_probability'] >= 0.5:
    print('Send email with promo')
else:
    print('dont do anything')