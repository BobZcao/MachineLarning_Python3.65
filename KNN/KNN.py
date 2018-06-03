from numpy import *
import operator 
import os
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDisIndicies = distances.argsort()
	classCount={}

	for i in range(k):
		voteIlabel = labels[sortedDisIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) +1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	#get the number of lines
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)

	#create a 0 filled matrix NumPy, Numpy is a two dimensional matrix
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0

	#use strip() to remove \return, get the elements by spliting using the \t
	#
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		if listFromLine[-1] == "largeDoses":
			classLabelVector.append(int(3))
		elif listFromLine[-1] == "smallDoses":
			classLabelVector.append(int(2))
		else:
			classLabelVector.append(int(1))
		index += 1
	return returnMat, classLabelVector

def autoNorm(dataSet):
	#get the row with min and max
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	#return row number of dataset
	m = dataSet.shape[0]
	#do the normalization, tile copy the same number of row of min and range
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	return normDataSet, ranges, minVals

def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat,ranges, minVals = autoNorm(datingDataMat)
	#number of row
	m = normMat.shape[0]
	numTestVecs = int(m * hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print("the classifier came back with: {}, the real answer is: {}".format(classifierResult,datingLabels[i]))
		if(classifierResult != datingLabels[i]): errorCount +=1
	print(f"the total error rate is: {(errorCount/float(numTestVecs)):.2f}")

def classifyPerson():
	resultList = ['Not at all','in small doeses','in large doeses']
	percentTats = float(input("percentage of time spend playing video games?"))
	ffMiles = float(input("frequent filer miles earned per year?"))
	iceCream = float(input("liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
	print ("You will probably like this person: {}".format(resultList[classifierResult-1]))	

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32 * i +j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = os.listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/{}'.format(fileNameStr))
	testFileList = os.listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/{}'.format(fileNameStr))
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("the classifier came back with:{}, the real answer is:{}".format(classifierResult, classNumStr))
		if(classifierResult != classNumStr): errorCount += 1.0
	print("\nthe total number of errors is: {}".format(errorCount))
	print("\nthe total error rate is: {}".format(errorCount/float(mTest)))
