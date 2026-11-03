FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -e . fastapi uvicorn
RUN python 01_Supervised_Learning/Regression/01_Linear_Regression/train.py
ENV PYTHONPATH=/app MODEL_PATH=/app/01_Supervised_Learning/Regression/01_Linear_Regression/models/model.joblib
EXPOSE 8000
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
