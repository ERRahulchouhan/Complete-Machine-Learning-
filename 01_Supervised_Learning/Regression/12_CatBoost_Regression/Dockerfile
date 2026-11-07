FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -e . catboost
ENV PYTHONPATH=/app
CMD ["python", "01_Supervised_Learning/Regression/12_CatBoost_Regression/train.py"]
