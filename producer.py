import json
import time
import logging
from kafka import KafkaProducer
from pyspark.ml.feature import MinMaxScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType

# Define the schema for the CSV file
schema = StructType([
    StructField("state", StringType(), True),
    StructField("account_length", LongType(), True),
    StructField("area_code", LongType(), True),
    StructField("international_plan", StringType(), True),
    StructField("voice_mail_plan", StringType(), True),
    StructField("number_vmail_messages", LongType(), True),
    StructField("total_day_minutes", DoubleType(), True),
    StructField("total_day_calls", LongType(), True),
    StructField("total_day_charge", DoubleType(), True),
    StructField("total_eve_minutes", DoubleType(), True),
    StructField("total_eve_calls", LongType(), True),
    StructField("total_eve_charge", DoubleType(), True),
    StructField("total_night_minutes", DoubleType(), True),
    StructField("total_night_calls", LongType(), True),
    StructField("total_night_charge", DoubleType(), True),
    StructField("total_intl_minutes", DoubleType(), True),
    StructField("total_intl_calls", LongType(), True),
    StructField("total_intl_charge", DoubleType(), True),
    StructField("customer_service_calls", LongType(), True),
    StructField("churn", StringType(), True)
])

# Create the SparkSession
spark = SparkSession.builder.getOrCreate()

def process_and_send_data():
    data_path = "churn-bigml-20.csv"
    df = spark.read.format('csv').option('header', True).schema(schema).load(data_path)
    df = df.withColumnRenamed("churn", "label")
    df = df.toDF(*(c.lower().replace(' ', '_') for c in df.columns))

    # Define transformers
    indexer = StringIndexer(inputCols=['state', 'international_plan', 'voice_mail_plan', 'label'],
                            outputCols=['state_indexed', 'international_plan_indexed', 'voice_mail_plan_indexed', 'label_indexed'])

    inputs = ['account_length', 'area_code', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
              'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
              'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
              'customer_service_calls']

    assembler1 = VectorAssembler(inputCols=inputs, outputCol="features_temp")
    scaler = MinMaxScaler(inputCol="features_temp", outputCol="features_scaled")
    assembler2 = VectorAssembler(inputCols=['state_indexed', 'international_plan_indexed', 'voice_mail_plan_indexed', 'features_scaled'], outputCol="features")

    # Apply transformers
    df = indexer.fit(df).transform(df)
    df = assembler1.transform(df)
    df = scaler.fit(df).transform(df)
    df = assembler2.transform(df)

    # Initialize Kafka producer
    producer = KafkaProducer(bootstrap_servers=['kafka:9093'], max_block_ms=5000, api_version=(2, 5, 0))

    # Send data to Kafka topic
    for row in df.collect():
        row_dict = {
            "state": row.state,
            "account_length": row.account_length,
            "area_code": row.area_code,
            "international_plan": row.international_plan,
            "voice_mail_plan": row.voice_mail_plan,
            "number_vmail_messages": row.number_vmail_messages,
            "total_day_minutes": row.total_day_minutes,
            "total_day_calls": row.total_day_calls,
            "total_day_charge": row.total_day_charge,
            "total_eve_minutes": row.total_eve_minutes,
            "total_eve_calls": row.total_eve_calls,
            "total_eve_charge": row.total_eve_charge,
            "total_night_minutes": row.total_night_minutes,
            "total_night_calls": row.total_night_calls,
            "total_night_charge": row.total_night_charge,
            "total_intl_minutes": row.total_intl_minutes,
            "total_intl_calls": row.total_intl_calls,
            "total_intl_charge": row.total_intl_charge,
            "customer_service_calls": row.customer_service_calls,
            "features": row.features.tolist(),
            "label_indexed": row.label_indexed
        }
        json_data = json.dumps(row_dict)  # Convert the row to JSON format
        print(json_data)
        producer.send('churn_topic', json_data.encode('utf-8'))
        time.sleep(3)  # Add a small delay between sending each row

    producer.close()

if __name__ == "__main__":
    try:
        process_and_send_data()
    except Exception as e:
        logging.error(f'An error occurred: {e}')
