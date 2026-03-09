# Databricks notebook source
# MAGIC %md
# MAGIC ##Bronze Layer

# COMMAND ----------

from pyspark.sql.functions import *

# Step 1: Read new Parquet files and add filename
df_new = spark.read.format("parquet").load("/mnt/input-data/dbx-proj/") \
    .withColumn("source_file", input_file_name())
df_new.display()

# COMMAND ----------



# Step 2: Read previously processed file names (if table exists)
try:
    df_existing = spark.read.format("delta").load("/mnt/input-data/delta/bronze/yellow_taxi")
    existing_files = [row.source_file for row in df_existing.select("source_file").distinct().collect()]
    df_new = df_new.filter(~col("source_file").isin(existing_files))  # Load only new files
except:
    print("No existing bronze table found. This is the initial load.")

df_new.display()
# Step 3: Append only new data
df_new.write.format("delta").mode("append").save("/mnt/input-data/delta/bronze/yellow_taxi")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Silver Layer

# COMMAND ----------

from pyspark.sql.functions import *

# When writing Bronze, include ingest timestamp
df_new = df_new.withColumn("ingest_time", current_timestamp())

# In Silver: Load only newly ingested records
df_bronze = spark.read.format("delta").load("/mnt/input-data/delta/bronze/yellow_taxi")
df_silver_existing = None

try:
    df_silver_existing = spark.read.format("delta").load("/mnt/delta/silver/yellow_taxi_enhanced")
    last_ingest_time = df_silver_existing.agg({"ingest_time": "max"}).collect()[0][0]
    df_bronze = df_bronze.filter(col("ingest_time") > lit(last_ingest_time))
except:
    print("Initial run of silver layer.")

# Now proceed with cleaning/enrichment on df_bronze...


# COMMAND ----------

# Load taxi zone lookup
df_zone = spark.read.csv("/mnt/input-data/dbx-proj-csv/taxi_zone_lookup.csv", header=True)
df_zone.display()

# COMMAND ----------

# Step 1: Clean key columns
df_clean = df_bronze.filter(
    (col("passenger_count").isNotNull()) & 
    (col("trip_distance").isNotNull()) &
    (col("fare_amount").isNotNull()) &
    (col("tpep_pickup_datetime").isNotNull()) &
    (col("tpep_dropoff_datetime").isNotNull())
)
df_clean.display()

# COMMAND ----------

# Step 2: Remove outliers
df_clean = df_clean.filter(
    (col("trip_distance") > 0) & (col("trip_distance") < 100) &    # Remove unrealistic long trips
    (col("fare_amount") > 0) & (col("fare_amount") < 500) &       # Remove crazy high fare values
    (col("passenger_count") <= 6)                                 # Cap realistic passenger count
)
df_clean.display()

# COMMAND ----------

# Step 3: Join with PU/DO zone details
df_zone = df_zone.withColumnRenamed("LocationID", "LocationID")  # Reassign to prevent collision

df_enriched = df_clean \
    .join(df_zone.withColumnRenamed("LocationID", "PULocationID"), "PULocationID", "left") \
    .withColumnRenamed("Borough", "PUBorough").withColumnRenamed("Zone", "PUZone") \
    .join(df_zone.withColumnRenamed("LocationID", "DOLocationID"), "DOLocationID", "left") \
    .withColumnRenamed("Borough", "DOBorough").withColumnRenamed("Zone", "DOZone")
df_enriched.display()

# COMMAND ----------

# Step 4: Derived Columns
df_transformed = df_enriched.withColumn(
    "trip_duration_minutes",
    (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60
)

# COMMAND ----------

# Step 5: Further cleaning on derived values
df_transformed = df_transformed.filter((col("trip_duration_minutes") > 1) & (col("trip_duration_minutes") < 180))

# Step 6: Tip Percentage & Tipping Behavior
df_transformed = df_transformed.withColumn(
    "tip_percentage", when(col("fare_amount") > 0, round((col("tip_amount") / col("fare_amount")) * 100, 2)).otherwise(0)
)

df_transformed = df_transformed.withColumn(
    "tipping_behavior", when(col("tip_percentage") >= 20, "Generous")
                         .when(col("tip_percentage") >= 10, "Average")
                         .otherwise("Low")
)
df_transformed.display()

# COMMAND ----------

# Step 7: Categorize Trip Type
df_transformed = df_transformed.withColumn(
    "trip_type",
    when((col("trip_distance") < 2) & (col("trip_duration_minutes") < 10), "Short Ride")
    .when((col("PUZone").like("%Airport%")) | (col("DOZone").like("%Airport%")), "Airport Trip")
    .otherwise("Regular Ride")
)
df_transformed.display()

# COMMAND ----------

# Step 8: Rename and Select Final Columns
df_final = df_transformed.select(
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", 
    "trip_distance", "trip_duration_minutes", "PUZone", "DOZone", 
    "fare_amount", "tip_amount", "total_amount", "tip_percentage", 
    "tipping_behavior", "trip_type", "payment_type"
)


# Step 9: Save to Silver Layer
df_final.write.format("delta").mode("overwrite").option("mergeSchema", "true").save("/mnt/input-data/delta/silver/yellow_taxi_enhanced")


# COMMAND ----------

# MAGIC %md
# MAGIC ##Gold Layer

# COMMAND ----------

# Load silver data
df_silver = spark.read.format("delta").load("/mnt/input-data/delta/silver/yellow_taxi_enhanced")

from pyspark.sql.functions import hour, avg, round

# Average fare and trip distance per pickup zone
df_zone_stats = df_silver.groupBy("PUZone").agg(
    round(avg("fare_amount"), 2).alias("avg_fare"),
    round(avg("trip_distance"), 2).alias("avg_distance")
)
df_zone_stats.display()

# COMMAND ----------

# Peak pickup hours
df_peak_hours = df_silver.withColumn("pickup_hour", hour("tpep_pickup_datetime")) \
    .groupBy("pickup_hour").count().orderBy("count", ascending=False)
df_peak_hours.display()

# COMMAND ----------

# Tip percentage analysis
df_tip_analysis = df_silver.withColumn("tip_percentage", (col("tip_amount") / col("fare_amount")) * 100) \
    .groupBy("payment_type").agg(round(avg("tip_percentage"), 2).alias("avg_tip_pct"))
df_tip_analysis.display()

# COMMAND ----------





# Write to Delta Lake
df_zone_stats.write.format("delta").mode("overwrite").save("/mnt/input-data/delta/gold/zone_stats")
df_peak_hours.write.format("delta").mode("overwrite").save("/mnt/input-data/delta/gold/peak_hours")
df_tip_analysis.write.format("delta").mode("overwrite").save("/mnt/input-data/delta/gold/tip_analysis")
