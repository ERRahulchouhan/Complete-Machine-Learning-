FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -e . xgboost
ENV PYTHONPATH=/app
CMD ["python", "01_Supervised_Learning/Regression/10_XGBoost_Regression/train.py"]
