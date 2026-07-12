# Heart Disease Prediction — MLOps Pipeline

## Setup
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
```bash
python data/download_data.py
python data/convert_to_csv.py
```

## Train
```bash
python src/train.py
```
Trained models land in `models/`, MLflow run data in `mlruns/`. View experiments:
```bash
mlflow ui --backend-store-uri mlruns
```

## Run tests
```bash
pytest tests/ -v
```

## Run the API
```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```
Test it:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
  "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
  "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3
}'
```

## Full stack (API + Prometheus + Grafana)
```bash
docker-compose up --build
```
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Kubernetes deployment
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get svc heart-disease-api-service
```

## Architecture
See `report.pdf` / `screenshots/` for the architecture diagram and deployment evidence.