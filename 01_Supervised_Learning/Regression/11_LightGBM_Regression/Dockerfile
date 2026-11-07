FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -e . lightgbm
ENV PYTHONPATH=/app
CMD ["python", "01_Supervised_Learning/Regression/11_LightGBM_Regression/train.py"]
