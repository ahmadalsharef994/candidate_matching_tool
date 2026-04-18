install:
	pip install -r requirements.txt

run:
	python app/main.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

match:
	python match_resumes.py

.PHONY: install run docker-up docker-down match
