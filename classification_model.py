from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("ml").getOrCreate()
schema = StructType([
    StructField("type", StringType(), False),
    StructField("sms", StringType(), False),
])

df = spark.read.csv("/user/hadoop/SMSSpamCollection", header=False, schema=schema, sep="\t")
df.show(10)

# Step 1: Convert the target variable into numerical labels using StringIndexer
indexer = StringIndexer(inputCol="type", outputCol="type_index")
df = indexer.fit(df).transform(df)

# Step 2: Tokenize and remove stopwords using a PySpark ML Pipeline
tokenizer = Tokenizer(inputCol="sms", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")

pipeline = Pipeline(stages=[tokenizer, remover, vectorizer])
model = pipeline.fit(df)
df = model.transform(df)

# Now, df contains numerical labels and tokenized features
# Display the transformed DataFrame
df.select("type", "type_index", "features").show(truncate=False)

# Split the data into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Create a Naive Bayes model
nb = NaiveBayes(featuresCol="features", labelCol="type_index", predictionCol="prediction")

# Train the model
nb_model = nb.fit(train_df)

# Make predictions on the test set
predictions = nb_model.transform(test_df)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="type_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Display the accuracy of the Naive Bayes model
print(f"Accuracy: {accuracy:.2%}")

predictions.select("type", "type_index", "prediction", "probability", "filtered_words", "sms").show(truncate=False)


# Stop the Spark session
spark.stop()
