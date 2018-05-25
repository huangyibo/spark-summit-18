from pyspark import SparkContext, SparkConf
import numpy as np
import keras
from keras.models import Sequential
import keras.layers as ll
import pyarrow as pa
from keras.models import model_from_json
import time



class mnist_loader(object):

    def __init__(self):
        print("nuet")
        
    
    def load(self, idx):
        startidx = idx[0]
        endidx = idx[1]
        print("Loading files from idx ",startidx," to idx ",endidx,"...")
        #fs = pa.hdfs.connect('flex02', 9000, user='jpf', driver='libhdfs')
        fs = pa.hdfs.connect()

        f = fs.open('/train-images-idx3-ubyte')
        f.seek(startidx*28*28+16)
        trainraw = f.read((endidx - startidx) * 28*28)
        f.close()
        trainbytes = bytearray(trainraw)
        
        t = []
        for i in xrange(0, endidx - startidx):
            subarray = np.array([trainbytes[i*28*28:(i+1)*28*28]]).reshape(28,28)
            subarray = subarray / 255.0
            t.append(subarray)
        X_train = np.array(t)
        
        f = fs.open('/train-labels-idx1-ubyte')
        f.seek(startidx+8)
        trainraw = f.read(endidx - startidx)
        f.close()
        trainbytes = bytearray(trainraw)
        
        l = []
        for i in xrange(0, endidx - startidx):
            subarray = np.zeros(10).reshape(10)
            subarray[trainbytes[i]] = 1.0
            l.append(subarray)
        y_train = np.array(l)
        
        
        
        trainraw = fs.cat('/t10k-images-idx3-ubyte')
        trainbytes = bytearray(trainraw)
        
        t = []
        for i in xrange(0, 10000):
            subarray = np.array([trainbytes[i*28*28+16:(i+1)*28*28+16]]).reshape(28,28)
            subarray = subarray / 255.0
            t.append(subarray)
        X_test = np.array(t)
        
        trainraw = fs.cat('/t10k-labels-idx1-ubyte')
        trainbytes = bytearray(trainraw)
        
        l = []
        for i in xrange(0, 10000):
            subarray = np.zeros(10).reshape(10)
            subarray[trainbytes[i+8]] = 1.0
            l.append(subarray)
        y_test = np.array(l)
        
        return(X_train, y_train, X_test, y_test)

def runNetwork(idx, weights, model_json):

    loader = mnist_loader()
    X_train, y_train, X_test, y_test = loader.load(idx)

    #the network
    model = model_from_json(model_json)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    model.summary()
    
    model.set_weights(weights)

    model.fit(X_train, y_train,
          validation_data=(X_test, y_test), epochs=1);
    
    return model.get_weights();


if __name__ == "__main__":

  # create Spark context with Spark configuration
  conf = SparkConf().setAppName("Spark Count")
  sc = SparkContext(conf=conf)

  loader = mnist_loader()
  X_train, y_train, X_test, y_test = loader.load((0,1))

  #the network
  model = Sequential(name="mlp")

  model.add(ll.InputLayer([28, 28]))
  model.add(ll.Flatten())
  model.add(ll.Dense(50))
  model.add(ll.Activation('sigmoid'))
  model.add(ll.Dense(10, activation='softmax'))

  model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

  model.summary()
  
  model_json = model.to_json()
  weights=model.get_weights()

  tasks = 4
  idx = []
  part = 60000 / tasks
  for i in xrange(tasks):
    idx.append((part * i, part * (i+1)))
#  idx = [(0,7500),(7500,15000),(15000,30000),(30000,45000),(45000,60000)]
  print(str(idx))
  idxp = sc.parallelize(idx,tasks)
  
  for i in xrange(30):
    o=idxp.map(lambda x:runNetwork(x, weights, model_json)).collect()
    npo=np.array(o)
    print "Fertig."
  
    average_weights=np.average(npo,axis=0)
    weights = average_weights.tolist()
    
    model.set_weights(weights)
    metrics = model.evaluate(X_test, y_test)
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))


  print " \n\nDone with training. Summary:\n"
  for i in range(len(model.metrics_names)):
      print(str(model.metrics_names[i]) + ": " + str(metrics[i]))

  print "\n\n"

