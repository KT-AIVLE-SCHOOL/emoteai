FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt /app/

RUN apt-get update && apt-get install -y build-essential

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY ./server_app /app

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]