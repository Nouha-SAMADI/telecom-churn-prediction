import logging
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, StructField, DoubleType, ArrayType, LongType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from cassandra.cluster import Cluster
import uuid


spark_session = SparkSession.builder \
    .appName('SparkDataStreaming') \
    .config('spark.jars.packages', "com.datastax.spark:spark-cassandra-connector_2.12:3.4.0,"
                                   "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .config('spark.cassandra.connection.host', 'cassandra:9042') \
    .getOrCreate()

def create_keyspace(session):
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS churn_stream
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'};
    """)

    print("Keyspace created successfully!")


def create_table(session):
    session.execute("""
    CREATE TABLE IF NOT EXISTS churn_stream.churn_predictions (
        user_id UUID PRIMARY KEY,
        state TEXT,
        account_length INT,
        area_code INT,
        international_plan TEXT,
        voice_mail_plan TEXT,
        number_vmail_messages INT,
        total_day_minutes DOUBLE,
        total_day_calls INT,
        total_day_charge DOUBLE,
        total_eve_minutes DOUBLE,
        total_eve_calls INT,
        total_eve_charge DOUBLE,
        total_night_minutes DOUBLE,
        total_night_calls INT,
        total_night_charge DOUBLE,
        total_intl_minutes DOUBLE,
        total_intl_calls INT,
        total_intl_charge DOUBLE,
        customer_service_calls DOUBLE,
        predicted_label DOUBLE,
        label_churn DOUBLE
    );
    """)

    print("Table created successfully!")


def insert_data(batch_df, batch_id):
    print("inserting data for batch:", batch_id)


    for row in batch_df.rdd.collect():
        user_id = uuid.uuid4()
        state = row['state']
        account_length = row['account_length']
        area_code = row['area_code']
        international_plan = row['international_plan']
        voice_mail_plan = row['voice_mail_plan']
        number_vmail_messages = row['number_vmail_messages']
        total_day_minutes = row['total_day_minutes']
        total_day_calls = row['total_day_calls']
        total_day_charge = row['total_day_charge']
        total_eve_minutes = row['total_eve_minutes']
        total_eve_calls = row['total_eve_calls']
        total_eve_charge = row['total_eve_charge']
        total_night_minutes = row['total_night_minutes']
        total_night_calls = row['total_night_calls']
        total_night_charge = row['total_night_charge']
        total_intl_minutes = row['total_intl_minutes']
        total_intl_calls = row['total_intl_calls']
        total_intl_charge = row['total_intl_charge']
        customer_service_calls = row['customer_service_calls']
        predicted_label = row['predicted_label']
        label_churn = row['label_indexed']



        try:
            session.execute("""
                INSERT INTO churn_stream.churn_predictions(user_id, state, account_length, area_code, international_plan, 
                    voice_mail_plan, number_vmail_messages, total_day_minutes, total_day_calls, total_day_charge,
                    total_eve_minutes, total_eve_calls, total_eve_charge, total_night_minutes, total_night_calls,
                    total_night_charge, total_intl_minutes, total_intl_calls, total_intl_charge, customer_service_calls,
                    predicted_label,label_churn)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
            """, (
            user_id, state, account_length, area_code, international_plan, voice_mail_plan, number_vmail_messages,
            total_day_minutes, total_day_calls, total_day_charge, total_eve_minutes, total_eve_calls,
            total_eve_charge, total_night_minutes, total_night_calls, total_night_charge, total_intl_minutes,
            total_intl_calls, total_intl_charge, customer_service_calls, predicted_label,label_churn))

            logging.info(f"Data inserted for {user_id}")

        except Exception as e:
            logging.error(f'could not insert data due to {e}')





def connect_to_kafka(spark_conn):
    spark_df = None
    try:
        spark_df = spark_conn.readStream \
            .format('kafka') \
            .option('kafka.bootstrap.servers', 'kafka:9093') \
            .option('subscribe', 'churn_topic') \
            .option('startingOffsets', 'earliest') \
            .load()
        logging.info("Kafka dataframe created successfully")
        return spark_df
    except Exception as e:
        logging.warning(f"Kafka dataframe could not be created because: {e}")
        return None


def create_cassandra_connection():
    try:
        # connecting to the Cassandra cluster
        cluster = Cluster(['cassandra'])

        cas_session = cluster.connect()

        return cas_session
    except Exception as e:
        logging.error(f"Could not create Cassandra connection due to {e}")
        return None


rf_model = RandomForestClassificationModel.load("pre_trained_model")


to_vector_udf = udf(lambda features: Vectors.dense(features), VectorUDT())
def create_selection_df_from_kafka(spark_df):
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
        StructField("customer_service_calls", DoubleType(), True),
        StructField("features", ArrayType(DoubleType()), True),
        StructField("label_indexed", DoubleType(), True)


    ])

    # Convertir les données JSON en colonnes
    sel = spark_df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col('value'), schema).alias('data')).select("data.*")


    # Convertir la colonne 'features' en vecteur
    sel = sel.withColumn("features", to_vector_udf(sel["features"]))

    return sel


def apply_prediction(df):
    # Appliquer la prédiction sur les caractéristiques de chaque ligne
    return rf_model.transform(df).withColumnRenamed("prediction", "predicted_label")


if __name__ == "__main__":
    # create Spark connection
    spark_conn = spark_session

    if spark_conn is not None:
        # Load RandomForestClassificationModel
        rf_model = RandomForestClassificationModel.load("pre_trained_model")

        # connect to Kafka with Spark connection
        spark_df = connect_to_kafka(spark_conn)
        selection_df = create_selection_df_from_kafka(spark_df)


        session = create_cassandra_connection()

        if session is not None:
            create_keyspace(session)
            create_table(session)

            logging.info("Streaming is being started...")

            streaming_query = (selection_df
                               .transform(apply_prediction)  # Appliquer la prédiction
                               .writeStream.format("org.apache.spark.sql.cassandra")
                               .foreachBatch(insert_data)  # Insérer les données dans Cassandra
                               .option("checkpointLocation", "/tmp/checkpoint")
                               .option("keyspace", "churn_stream")  # Ajouter keyspace
                               .option("table", "churn_predictions")  # Ajouter table
                               .start())

            streaming_query.awaitTermination()