// 
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.feature.StandardScaler

val t1 = System.nanoTime

val data_rdd = sc.textFile("/home/ubuntulab/Downloads/gearbox_readings/archive")//.sample(false,0.01)


//val data_raw = spark.read.csv("edicare_Part_D_Prescribers_by_Provider_and_Drug_2019.csv/Run_*")

//val data_rdd = sc.textFile("gearbox_readings/archive")

val labelsAndData = data_rdd.map { line =>
  val buffer = line.split(',').toBuffer
  val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
  (vector)
}

val data = labelsAndData.cache()

/////////////////////scaling the data//////////////////////////

//val scaler = new StandardScaler(true, true).fit(data)
//data.foreach{ vector =>
          //val scaled = scaler.transform(vector)
 //}

//////////////////////Normalizing the data/////////////////////

import org.apache.spark.sql.functions


def normalize(data: RDD[Vector]): (Vector => Vector) = {
  val dataAsArray = data.map(_.toArray)
  val numCols = dataAsArray.first().length
  val n = dataAsArray.count()
  val sums = dataAsArray.reduce(
    (a, b) => a.zip(b).map(t => t._1 + t._2))
  val sumSquares = dataAsArray.aggregate(
    new Array[Double](numCols))(
      (a, b) => a.zip(b).map(t => t._1 + t._2 * t._2),
      (a, b) => a.zip(b).map(t => t._1 + t._2))
  val stdevs = sumSquares.zip(sums).map {
    case (sumSq, sum) => math.sqrt(n * sumSq - sum * sum) / n
  }
  val means = sums.map(_ / n)

  (vector: Vector) => {
    val normalizedArray = (vector.toArray, means, stdevs).zipped.map(
      (value, mean, stdev) =>
        if (stdev <= 0) (value - mean) else (value - mean) / stdev)
    Vectors.dense(normalizedArray)
  }
}

//////////////////////////////////////////////////////////////////

val normalizedData = data.map(normalize(data)).cache()
normalizedData.cache()


//////////running K mean on normalized data//////////////////////

val kmeans = new KMeans()
kmeans.setK(10)
val model = kmeans.run(normalizedData)
val sample = normalizedData.map(vector =>
model.predict(vector) + "," + vector.toArray.mkString(",")
)//.sample(false, 0.006)
//sample.saveAsTextFile("./kmeans-sample")



/////Investigate the Average Distance to Closest Centroid//////// 

def distance(a: Vector, b: Vector) =
  math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d * d).sum)

def distToCentroid(vector: Vector, model: KMeansModel) = {
  val cluster = model.predict(vector)
  val centroid = model.clusterCenters(cluster)
  distance(centroid, vector)
}



////////////////And Run the Final Anomaly Detection Strategy////////

def anomaly(data: RDD[Vector],
  normalizeFunction: (Vector => Vector)): (Vector => Boolean) = {val normalizedData = data.map(normalizeFunction)
  normalizedData.cache()
  val kmeans = new KMeans()
  kmeans.setK(10)
  val model = kmeans.run(normalizedData)

  normalizedData.unpersist()

  val distances = normalizedData.map(vector => distToCentroid(vector, model))
  val threshold = distances.top(100).last

  (vector: Vector) => distToCentroid(normalizeFunction(vector), model) > threshold
}


val originalAndData = data.map(line => (line, line))
val anomalyFunction = anomaly(data, normalize(data))
val anomalies = originalAndData.filter {
  case (line, vector) => anomalyFunction(vector)
}.keys

//anomalies.saveAsTextFile("./anomalies-sample")

//anomalies.take(10).foreach(println)



val duration = (System.nanoTime - t1) / 1e9d


