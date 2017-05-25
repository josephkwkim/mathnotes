import neuralNet as neural
import unpackData as unpack
import numpy as np
import time

def testAccuracy(net,trainX,trainY):
    n = len(trainY)
    count = 0
    for index in range(len(trainX)):
        output = np.argmax(net.feedforward(np.reshape(trainX[index],(1024,1))))
        expected = np.argmax(trainY[index])
        count += (output==expected)
    return count/n

def trainNet(netSize,trainX,trainY,epochs):
    np.seterr(all="raise")
    print ("loaded data")
    net=neural.Network(netSize)
    print ("built net")
    a=time.time()
    previousAccuracy = testAccuracy(net,trainX,trainY)
    for i in range(epochs):
        currentAccuracy = testAccuracy(net,trainX,trainY)
        print("Accuracy before run", i, ": ", currentAccuracy)
        if currentAccuracy-previousAccuracy<=0:
            neural.self.learningRate/=2
        net.MBGD(trainX,trainY,50)
        print ("Time after run",i,": ",(time.time()-a))
    return net


#TRAIN_X = unpack.loadData()[0][0]
#TRAIN_Y = unpack.loadData()[0][1]
#TEST_X = unpack.loadData()[1][0]
#TEST_Y = unpack.loadData()[1][1]
#TRAINED_NET = trainNet([1024,39,369],TRAIN_X,TRAIN_Y,2)

### FOR DEBUGGING ONLY ###
#DEBUG_X = TEST_X[:1500]
#DEBUG_Y = TEST_Y[:1500]
#DEBUG_NET = trainNet([1024,400,369],DEBUG_X,DEBUG_Y,5)

def saveData():
    np.savetxt("debuggingX",DEBUG_X)
    np.savetxt("debuggingY",DEBUG_Y)
    print ("saved files")

def loadTestingData():
    debug_x = np.loadtxt("debuggingX")
    debug_y = np.loadtxt("debuggingY")
    return debug_x,debug_y

def scaletestData(x):
    for i in range(len(x)):
        for value in range(len(x[i])):
            if x[i][value]==255:
                x[i][value]=0
            else:
                x[i][value]=1
    return x

def test():
    netsize = [1024,400,369]
    trainx = scaletestData(loadTestingData()[0])
    trainy = loadTestingData()[1]
    trainNet(netsize,trainx,trainy,100)
    print ("Done!")

def testRealShit():
    trainX = scaletestData(TRAIN_X)
    trainY = TRAIN_Y
    netsize = [1024,200,369]
    trainNet(netsize,trainX,trainY,20)
    print ("You Fucked Up!!!")




