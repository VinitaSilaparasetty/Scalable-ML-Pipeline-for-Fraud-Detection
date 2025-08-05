"""
Scalable ML Pipeline: Fraud Detection using Spark and MLflow
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
import mlflow

# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetection").getOrCreate()

# Load dataset
df = spark.read.csv("transactions.csv", header=True, inferSchema=True)

# Feature assembly
features = [col for col in df.columns if col not in ['label']]
assembler = VectorAssembler(inputCols=features, outputCol="features")
data = assembler.transform(df).select("features", "label")

# Train model
model = RandomForestClassifier(labelCol="label", featuresCol="features").fit(data)

# Log model with MLflow
mlflow.set_experiment("FraudDetection")
with mlflow.start_run():
    mlflow.spark.log_model(model, "rf-model")

print("Model trained and logged to MLflow.")
