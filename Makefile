install:
	python -m pip install -r requirements.txt

train:
	python -m src.train

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-build:
	docker build -t churn-api .

docker-run:
	docker run -p 8000:8000 -e CHURN_API_KEY=change-me churn-api
