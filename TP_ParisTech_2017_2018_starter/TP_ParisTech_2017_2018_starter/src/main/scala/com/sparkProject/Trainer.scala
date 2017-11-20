package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /** CHARGER LE DATASET **/
    val df: DataFrame = spark
      .read
      .parquet("../prepared_trainingset")

    df.show()
    df.printSchema()

    /** TF-IDF **/
      /** Stage 1 **/
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

      /** Stage 2 **/
    val stopWordSet = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("tokens_cl")

      /** Stage 3 **/
    val vectorizer = new CountVectorizer()
      .setVocabSize(50)
      .setInputCol("tokens_cl")
      .setOutputCol("vectorized")

      /** Stage 4 **/
    val idfer = new IDF()
      .setInputCol("vectorized")
      .setOutputCol("tfidf")

      /** Stage 5 **/
    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")

      /** Stage 6 **/
    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")

    /** VECTOR ASSEMBLER **/
    val cols = Array("tfidf", "days_campaign", "hours_prepa",
      "goal", "country_indexed", "currency_indexed")
    val vecAssembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")

    /** MODEL **/
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** PIPELINE **/
    val stages = Array(tokenizer, stopWordSet,
      vectorizer, idfer, countryIndexer, currencyIndexer,
      vecAssembler, lr)
    val pipe = new Pipeline().setStages(stages)

    /** TRAINING AND GRID-SEARCH **/
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1),
      seed=15)

    val grid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(vectorizer.minDF, Array(55.0, 75.0, 95.0))
      .build()

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val tvSplit = new TrainValidationSplit()
      .setEstimator(pipe)
      .setEvaluator(eval)
      .setEstimatorParamMaps(grid)
      .setTrainRatio(0.7)

    val model = tvSplit.fit(training)

    val df_WithPredictions = model.transform(test)

    val f1Score = eval.evaluate(df_WithPredictions)
    println("F1Score : " + f1Score)

    df_WithPredictions.groupBy("final_status", "predictions").count().show()

    model.save("../trainedModel")
  }
}
