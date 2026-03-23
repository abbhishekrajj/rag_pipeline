# 1. Base image
FROM python:3.11-slim

# 2. Work directoryFROM python:3.14-s
WORKDIR /app

# 3. Dependencies copy aur install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Saara code copy karein
COPY . .

# 5. Port expose karein
EXPOSE 8000

# 6. FastAPI server start karein
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

#CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]