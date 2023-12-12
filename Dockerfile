FROM python:3.10.12

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r  requirements.txt

COPY . .

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

# CMD ["python", "-u", "main.py"]

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]