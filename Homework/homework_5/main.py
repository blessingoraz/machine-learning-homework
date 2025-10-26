# Server

import pickle

import uvicorn

from fastapi import FastAPI

#Load data
with open('pipeline_v1.bin', 'rb') as f_in: # Loading or reading
    pipeline = pickle.load(f_in)

app = FastAPI(title="Lead Score Prediction")

def predict_leadscore(data):
    leadscore = pipeline.predict_proba([data])[0, 1]  # probability of lead conversion

    print(f'Predicted lead score: {leadscore:.4f}')
    return float(leadscore)

# ----- endpoint -----
@app.post("/score")
def score_lead(data: dict):
    leadscore = predict_leadscore(data)
    return {"lead_score": leadscore}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)