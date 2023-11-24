FROM python:3.9-slim-bullseye

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app/
COPY main.py /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 32168

CMD ["python3", "main.py", "--host", "0.0.0.0", "--port", "32168"]
