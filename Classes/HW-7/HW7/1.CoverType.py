# Name: Xia Song
# Email: xis103@eng.ucsd.edu
# PID: A53228931

from pyspark import SparkContext
sc = SparkContext()
# coding: utf-8


from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils


# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
path='/covtype/covtype.data'
inputRDD=sc.textFile(path)

# Transform the text RDD into an RDD of LabeledPoints
Data=inputRDD.map(lambda line: [float(strip(x)) for x in line.split(',')]).map(lambda line: LabeledPoint(line[-1], line[0:-1]))
         

# ### Making the problem binary
# 
# The implementation of BoostedGradientTrees in MLLib supports only binary problems. the `CovTYpe` problem has
# 7 classes. To make the problem binary we choose the `Lodgepole Pine` (label = 2.0). We therefor transform the dataset to a new dataset where the label is `1.0` is the class is `Lodgepole Pine` and is `0.0` otherwise.

# In[40]:

Label=2.0
mapping = {Label: 1.0}
Data=inputRDD.map(lambda line: [float(x) for x in line.split(',')]).map(lambda V: LabeledPoint(mapping.setdefault(V[-1], 0.0), V[0:-1]))

# ### Reducing data size
# In order to see the effects of overfitting more clearly, we reduce the size of the data by a factor of 10

(trainingData,testData)=Data.randomSplit([0.7,0.3], seed=255)

from time import time
errors={}
for depth in [10]:
    start=time()
    model=GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo={}, 
                                               numIterations=10, maxDepth=depth, learningRate=0.25,
                                              maxBins=54)
    #print model.toDebugString()
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in dataSets.keys():  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=data.map(lambda line: line.label).zip(Predicted)
        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
        errors[depth][name]=Err
    print depth,errors[depth],int(time()-start),'seconds'