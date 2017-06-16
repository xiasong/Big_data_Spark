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
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils


# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
path='/HIGGS/HIGGS.csv'
inputRDD=sc.textFile(path)


# Transform the text RDD into an RDD of LabeledPoints
Data=inputRDD.map(lambda line: [float(strip(x)) for x in line.split(',')]).map(lambda line: LabeledPoint(line[0], line[1:]))

#sampling 1% data from dataset and split data into train and test data
Data1=Data.sample(False,0.1)
(trainingData,testData)=Data1.randomSplit([0.7,0.3], seed=255)


#Random Forest
from time import time
errors={}
for depth in [10]:
    start=time()
    model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=1, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=depth)
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in dataSets.keys():  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))

        LabelsAndPredictions = data.map(lambda lp: lp.label).zip(Predicted)
        Err = LabelsAndPredictions.filter(lambda (v,p): v != p).count()/float(data.count())
        errors[depth][name]=Err
    print depth,errors[depth],int(time()-start),'seconds'
