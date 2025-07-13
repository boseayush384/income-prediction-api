# Income Prediction API

A machine learning API that predicts whether an individual's income exceeds $50K using demographic data. Built using `FastAPI`, containerized with `Docker`, and includes unit tests.

---

## Features

- Predicts income level (`>50K` or `<=50K`) from 14 features
- Includes `pytest` unit test suite
- Dockerized app for easy deployment
- Built using a Logistic Regression pipeline (sklearn)

---

## Folder Structure

income-prediction-api/
│
├── app/
│ └── main.py # FastAPI app with prediction logic
│ └── model.pkl # Trained model artifact
│
├── data/
│ └── income_evaluation.csv # Source dataset
│
├── src/
│ ├── preprocess.py # Data preprocessing logic
│ ├── model.py # Model training and serialization
│
├── tests/
│ └── test_api.py # Unit test for /predict endpoint
│
├── Dockerfile # For building Docker image
├── train.py # Script to train and save model
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)

## Model Training & Saving

Train and persist the logistic regression model:

```bash
python train.py

## Serving with FastAPI

Run the FastAPI server locally:
uvicorn app.main:app --reload

Visit docs at: http://127.0.0.1:8000/docs

## Dockerized Deployment

Build and run the API inside a Docker container:
# Build the image
docker build -t income-api .

# Run the container
docker run -d -p 8000:8000 income-api

## Run Unit Tests

Run Pytest to validate /predict API response:
pytest -v tests/test_api.py

## Example API Request

Endpoint
POST /predict

Sample JSON Input

{
  "age": 37,
  "workclass": "Private",
  "fnlwgt": 284582,
  "education": "Masters",
  "education-num": 14,
  "marital-status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 15000,
  "capital-loss": 0,
  "hours-per-week": 60,
  "native-country": "United-States"
}

Sample Output
{
  "prediction": ">50K",
  "input": {
    "age": 37,
    ...
  }
}

Requirements
Install Python dependencies:

pip install -r requirements.txt

Author
Ayush Bose
Senior Data Scientist | GenAI | ML Engineering
LinkedIn - https://www.linkedin.com/in/boseayush384/
