FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app


RUN mkdir /app/models
COPY ["source/predict.py", "./"] 
COPY ["models/the_best_model.pkl", "./models"]


COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy


EXPOSE 8000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8000", "predict:app"]
