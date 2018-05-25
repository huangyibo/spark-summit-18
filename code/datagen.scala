import org.apache.spark.sql.{SaveMode, SparkSession}

import scala.collection.mutable.ListBuffer
import scala.util.Random

private def generateTSRecord(key: Array[Byte], recBuf:Array[Byte], rand: Random): Unit = {
  val fixed = 10
  // Generate the 10-byte key using the high 10 bytes of the 128-bit random number
  var i = 0
  while (i < 10) {
    key(i) = rand.nextInt().toByte
    i += 1
  }

  // Add 2 bytes of "break"
  recBuf(10 - fixed) = 0x00.toByte
  recBuf(11 - fixed) = 0x11.toByte

  // Convert the 128-bit record number to 32 bits of ascii hexadecimal
  // as the next 32 bytes of the record.
  i = 0
  while (i < 32) {
    recBuf(12 + i  - fixed) = rand.nextInt().toByte
    i += 1
  }

  // Add 4 bytes of "break" data
  recBuf(44  - fixed) = 0x88.toByte
  recBuf(45  - fixed) = 0x99.toByte
  recBuf(46  - fixed) = 0xAA.toByte
  recBuf(47  - fixed) = 0xBB.toByte

  // Add 48 bytes of filler based on low 48 bits of random number
  i = 0
  while (i < 12) {
    val v = rand.nextInt().toByte
    recBuf((48 + i * 4) - fixed) = v
    recBuf((49 + i * 4) - fixed) = v
    recBuf((50 + i * 4) - fixed) = v
    recBuf((51 + i * 4) - fixed) = v
    i += 1
  }

  // Add 4 bytes of "break" data
  recBuf(96 - fixed) = 0xCC.toByte
  recBuf(97 - fixed) = 0xDD.toByte
  recBuf(98 - fixed) = 0xEE.toByte
  recBuf(99 - fixed) = 0xFF.toByte
}

def generateTerasortData(cores:Int = 8, totalRecs:Long = 10 * 1000, parts:Int = 8, location:String = "/tmp/input"):Unit = {
  val LIMIT:Long = (1L << 31) / 100
  val keySize = 10
  val valSize = 90
  val spark:SparkSession = SparkSession.builder().getOrCreate()
  val perTask:Long = totalRecs / cores
  require((totalRecs % cores == 0) && (perTask < LIMIT), " ent:  " + totalRecs + " cores " + cores + " ent % cores " + (totalRecs % cores) + " perTask " + perTask + " LIMIT " + LIMIT)

  import spark.implicits._
  val inputJobRDD = spark.sparkContext.parallelize(0 until cores, cores)
  val recordInputDS = inputJobRDD.flatMap { p =>
    val base = new ListBuffer[Tuple2[Array[Byte], Array[Byte]]]()
    val rand = new Random(System.nanoTime())
    /* now we want to generate a loop and save the avro file */
    var i = 0L
    while (i < perTask) {
      val key = new Array[Byte](10)
      val value = new Array[Byte](90)
      generateTSRecord(key, value, rand)
      i += 1
      base += Tuple2(key, value)
    }
    base
  }.toDS().repartition(parts)

  import com.databricks.spark.avro._
  recordInputDS.write.mode(SaveMode.Overwrite).avro(location)
  val totalSize = totalRecs * (100)
  println("Wrote terasort file: " + location + " total count: " + totalRecs + " totalSize: " + totalSize)
}


private def generatePayloadRecord(payload: Array[Byte], rand: Random): Unit = {
  // we pick 4 random points instead of generating a whole new array which might be super slow
  val pivot = if(payload.length < 4) 1 else payload.length / 4
  payload(pivot) = rand.nextInt().toByte
  payload(pivot * 2) = rand.nextInt().toByte
  payload(pivot * 3) = rand.nextInt().toByte
  if(payload.length % 4 == 0)
    payload(pivot * 4 - 1) = rand.nextInt().toByte
  else
    payload(pivot * 4) = rand.nextInt().toByte
}


def generatePayloadDataSZ(cores:Int = 8, totalRecordSize:Long = 10 * 1000, parts:Int = 8, payloadSize:Int = 100, location:String = "/tmp/input") = {
  val recCount:Long = totalRecordSize / (payloadSize + 4)
  generatePayloadData(cores, recCount, parts, payloadSize, location)
}

def generatePayloadData(cores:Int = 8, totalRecs:Long = 10 * 1000, parts:Int = 8, payloadSize:Int = 100, location:String = "/tmp/input"):Unit = {
  val LIMIT:Long = 1L << 31 // 2GB max for DBFS
  val spark:SparkSession = SparkSession.builder().getOrCreate()
  val perTask:Long = totalRecs / cores
  val singleFileSize = perTask * (payloadSize + 4)
  require((totalRecs % cores == 0) && (singleFileSize < LIMIT), " ent:  " + totalRecs + " cores " + cores + " ent % cores " + (totalRecs % cores) + " singleFileSize " + singleFileSize + " LIMIT " + LIMIT)

  import spark.implicits._
  val inputJobRDD = spark.sparkContext.parallelize(0 until cores, cores)
  val recordInputDS = inputJobRDD.flatMap { p =>
    val base = new ListBuffer[Tuple2[Int, Array[Byte]]]()
    val rand = new Random(System.nanoTime())
    /* now we want to generate a loop and save the avro file */
    var i = 0L
    while (i < perTask) {
      val payload = new Array[Byte](payloadSize)
      generatePayloadRecord(payload, rand)
      i += 1
      base += Tuple2(rand.nextInt(), payload)
    }
    base
  }.toDS().repartition(parts)

  import com.databricks.spark.avro._
  recordInputDS.write.mode(SaveMode.Overwrite).avro(location)
  val totalSize = totalRecs * (payloadSize + 4)
  println("Wrote file: " + location + " total count: " + totalRecs + " payload: " + payloadSize + " totalSize: " + totalSize)
}


//README:
// There are two modes that you can use:
// generatePayloadData  : that generates (int, payload) schema
// generateTerasortData : that generates (10 + 90) bytes values
// they take following parameters:
// cores:Int = 8 (default): Number of cores to generate data
// totalRecord:Int = 10 * 1000 : Total number of records
// parts:Int = 8 : In how many parts should the data be saved
// payloadSize:Int = 100 : payload size for payload data schema
// location:String = "/tmp/input" - the location of the output

// How to run SQL query
// CREATE TABLE output LIKE input;
// INSERT INTO output SELECT * FROM input ORDER BY _1 ASC

// How to delete data from DBFS
// %fs rm -r /tmp/fooAvro (run this on any shell)
