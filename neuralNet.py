##Built-In Libraries##
import time
import csv
import string

##Third-Party Libraries##
import numpy as np
from PIL import Image,ImageTk
import wolframalpha

##Other TP Files##
import mnist_training as mnist #See File for Citations
import unpackData as unpack #See File for Citations

class Network(object):
    def __init__(self,other): #other is a list containing the number of
        #neurons per layer.  A net with three inputs, a two-node hidden layer,
        #and one output would be represented as [3,2,1]
        self.numLayers=len(other)
        self.netSize=other
        self.learningRate=0.3 #learning rate is the step size for gradient
        #descent.  The ideal learning rate varies by net.
        self.count = 0 #count interations for auto_stop
        self.biases=[np.random.randn(i,1)/32 \
                      for i in self.netSize[1:]]
        #this initializes a 2D list of random, normally distributed values
        #which represent the initial bias values for each node
        self.weights=[np.random.randn(b,a)/32 \
                      for (a,b) in zip(self.netSize[:-1],self.netSize[1:])]
        #initializes a 3D list of weights associated with each neuron.

    def save(self,title):
        f = open("WeightsMNIST"+str(title)+".txt","wb+")
        np.save(f,self.weights)
        g = open("BiasesMNIST"+str(title)+".txt","wb+")
        np.save(g,self.biases)
        print ("Saved!")

    def feedforward(self, a):
        #a is the input matrix of size (n,1) where n is the number of neurons
        #in the first row
        for index in range(len(self.netSize)-1):
            #np.dot performs matrix multiplication in 2D and regular dot
            #product in 1D assuming the dimensions are correct
            nextLayer=np.dot(self.weights[index],a)
            #nextLayer is the weighted sum of all the previous inputs
            #arranged as a vector based on how many neurons are in the
            #next row
            a=sigmoid(nextLayer+self.biases[index])
            #the sigmoid takes in the weighted array and adds the bias and
            #returns an array with the same dimensions just with modified
            #values
        return a

    def MBGD(self,trainX,trainY,batchsize=1,test_x=None,test_y=None):
        #print ("learning rate:",self.learningRate)
        if not isinstance(test_x,type(None)): #check if test data is provided
            before = testAccuracy(self,test_x,test_y)
        combinedData = list(zip(trainX,trainY)) #combine x,y for shuffling
        np.random.shuffle(combinedData) #randomly shuffle it
        shuffled = list(zip(*combinedData)) #unzip/separate it

        for batch in range(len(trainX)//batchsize): #goes through batches
            start = batch*batchsize #start of batch interval
            end = start+batchsize #end of batch interval (start-end=batchsize)
            updateW = [np.zeros(w.shape) for w in
                       self.weights]  # zero vector with
            # same shape as self.weights
            updateB = [np.zeros(b.shape) for b in
                       self.biases]  # zero vector with
            # same shape as self.biases
            for index in range(start,end): #loop through individual batch
                x = shuffled[0][index]
                y = shuffled[1][index]
                #entry = np.reshape(x,(1024,1))
                #exit = np.reshape(y,(94,1))
                gradB,gradW=self.backprop(x,y) #graident vectors with the same
                #shape as self.weights and self.biases with the gradients
                #of the cost function computed by the backpropgation algorithm
                updateW=[uw+gw for uw,gw in zip(updateW,gradW)]
                updateB=[ub+gb for ub,gb in zip(updateB,gradB)]

            for index in range(len(self.weights)): #update Weights
                weights = self.weights[index]
                update = (self.learningRate)*updateW[index]/batchsize
                newVal=weights-update
                self.weights[index]=newVal

            for index in range(len(self.biases)): #update Biases
                newVal=(self.biases[index]-
                        (self.learningRate*updateB[index]/batchsize))
                self.biases[index]=newVal

        if not isinstance(test_x,type(None)): #check if test data is provided
            after = testAccuracy(self,test_x,test_y)
            if after-before<=0: #if accuracy has gone down
                self.count+=1
                self.learningRate/=2 #lower the learning rate


###NOT MY CODE#### (backprop) #taken from Neural Networks and Deep Learning

            #book online by Michael Nielsen

###Not my code (Backprop)###

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = costDerivative(activations[-1], y) * \
            sigmoidPrime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.numLayers):
            z = zs[-l]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
###^^^^^^Not My Code^^^^^^### (backprop written by Michael Nielsen)

    def setNet(self):
        self.weights = [np.array([[1,2],[3,2],[1,2]]),np.array([[3,2,1]])]
        self.biases = [np.array([[1],[1],[1]]),np.array([[1]])]

###Math Functions###

def cost(actual,ideal): #takes in (n,1) and (n,1) arrays where actual is the
    #output of the network and #ideal is the ideal result. Returns MSE
    #according to the cost function
    MSE=0 #MSE is Mean Squared Error
    for index in range(len(actual)):
        errorSq=(actual[index][0]-ideal[index][0])**2
        MSE+=errorSq
    MSE/=(2*len(actual)) #WLOG we can use actual, we could also use ideal
    return MSE

def costDerivative(actual,ideal):
    return (actual-ideal)

from scipy.special import expit #expit is a built-in sigmoid function with high
#floating point arithmetic accuracy

def sigmoid(z):
    return expit(z)

def sigmoidPrime(z):
    return sigmoid(z)*(1-sigmoid(z))

###TESTING FUNCTIONS###

def testAccuracy(net,trainX,trainY):
    n = len(trainY)
    seen = []
    count = 0
    for index in range(len(trainX)):
        output = np.argmax(net.feedforward(np.reshape(trainX[index],(1024,1)))) #np.reshape for non-mnist
        #output = np.argmax(net.feedforward(trainX[index]))
        expected = np.argmax(trainY[index])
        #print ("Output:",output,"expected:",expected)
        count += (output==expected)
        if output==expected:
            if expected not in seen:
                seen.append(expected)
    return count/n

def trainNet(netSize,trainX,trainY,epochs,testx=1,testy=1):
    np.seterr(all="raise")
    net=Network(netSize)
    for i in range(epochs):
        if net.count==10:
            net.save(net.count-10)
            print ("accuracy has gone down and up 10 times")
        a=time.time()
        currentAccuracy = testAccuracy(net,trainX,trainY)
        if not isinstance(testx,int):
            otherAccuracy = testAccuracy(net,testx,testy)
        print("Accuracy before run", i, ":", currentAccuracy)
        if not isinstance(testx,int):
            print("Accuracy of Test Data:",otherAccuracy)
        net.MBGD(trainX,trainY,100,trainX,trainY)
        print ("Time after run",i,":",(time.time()-a))
    return net

def testSmallDataset(): #tests on small batch (1500) saved to txt file
    netsize = [1024,500,369]
    trainx = 1-(loadTestingData()[0]/255)
    trainy = loadTestingData()[1]
    net = trainNet(netsize,trainx,trainy,100)
    #print (net.feedforward(np.reshape(trainx[5],(1024,1))))
    print ("Done!")

def testReal():
    trnx,trny,tstx,tsty = TRAIN_X(),TRAIN_Y(),TEST_X(),TEST_Y()
    netsize = [1024,200,94]
    net = trainNet(netsize,trnx,trny,10,tstx,tsty)
    accuracy = testAccuracy(net,tstx,tsty)
    net.save("hasy")
    return ("Accuracy With Test Data: "+str(accuracy))

def testExistingHasy():
    trnx, trny, tstx, tsty = TRAIN_X(), TRAIN_Y(), TEST_X(), TEST_Y()
    net = createNet()
    for i in range(5):
        print ("accuracy before run",str(i),str(testAccuracy(net,tstx,tsty)))
        net.MBGD(trnx,trny,batchsize=100,test_x=tstx,test_y=tsty)
    accuracy = testAccuracy(net,tstx,tsty)
    net.save("hasy")
    print ("Accuracy with Test Data: "+str(accuracy))

def trainExistingMnist():
    net = createMNIST()
    train_x = scaletestData(mnist.load_data_wrapper()[0])
    train_y = mnist.load_data_wrapper()[1]
    test_x = scaletestData(mnist.load_data_wrapper()[2])
    test_y = mnist.load_data_wrapper()[3]
    for i in range(5):
        net.MBGD(train_x,train_y,batchsize=100,test_x=test_x,test_y=test_y)
        print ("accuracy after run",i,testAccuracy(net,test_x,test_y))
    net.save('JUSTDIDTHISTONIGHT')
    print ("saved and done")

#for HASY
def loadWeightsAndBiases():
    weights = np.load("WeightsMNISThasy.txt")
    biases = np.load("BiasesMNISThasy.txt")
    return list(weights),list(biases)
#for HASY
def createNet():
    net = Network([1024,200,94])
    net.weights=loadWeightsAndBiases()[0]
    net.biases=loadWeightsAndBiases()[1]
    return net
#for MNIST
def loadMnist():
    weights = np.load("WeightsMNISTJUSTDIDTHISTONIGHT.txt")
    biases = np.load("BiasesMNISTJUSTDIDTHISTONIGHT.txt")
    return list(weights),list(biases)
#for MNIST
def createMNIST():
    net = Network([784,100,10])
    net.weights = loadMnist()[0]
    net.biases = loadMnist()[1]
    return net

def savePic(datax,datay,index):
    arrayyy = datax[index]
    ind = np.argmax(datay[index])
    print ("index:",ind)
    print ("done!")

def mnistTest(epochs):
    netsize = [784, 100, 10]
    train_x = scaletestData(mnist.load_data_wrapper()[0])
    train_y = mnist.load_data_wrapper()[1]
    test_x = scaletestData(mnist.load_data_wrapper()[2])
    test_y = mnist.load_data_wrapper()[3]
    net = trainNet(netsize,train_x,train_y,epochs,test_x,test_y)
    net.save('JUSTDIDTHISTONIGHT')
    print ("saved and done!")

###CREATE AND LOAD TESTING DATA###

def TRAIN_X():
    x = scaletestData(unpack.loadData()[0][0])
    print (x[0])
    return x

def TRAIN_Y():
    return unpack.loadData()[0][1]

def TEST_X():
    return scaletestData(unpack.loadData()[1][0])

def TEST_Y():
    return unpack.loadData()[1][1]

def TRN_X():
    return setUpTestData()[0]

def TRN_Y():
    return setUpTestData()[1]

def TST_X():
    return setUpTestData()[2]

def TST_Y():
    return setUpTestData()[3]

def saveData():
    np.savetxt("debuggingX",DEBUG_X)
    np.savetxt("debuggingY",DEBUG_Y)
    print ("saved files")

def setUpTestData():
    testx = TEST_X()
    testy = TEST_Y()
    trn_x,trn_y,tst_x,tst_y=[],[],[],[]
    for i in range(len(testx)):
        if i%10==0:
            tst_x.append(testx[i])
            tst_y.append(testy[i])
        else:
            trn_x.append(testx[i])
            trn_y.append(testy[i])
    return trn_x,trn_y,tst_x,tst_y

def loadTestingData():
    debug_x = np.loadtxt("debuggingX")
    debug_y = np.loadtxt("debuggingY")
    return debug_x,debug_y

def scaletestData(x):
    zeros,ones = (0,0)
    for i in range(len(x)):
        for value in range(len(x[i])):
            if x[i][value]!=0:
                x[i][value]=0
                zeros+=1
            else:
                x[i][value]=1
                ones+=1
    if ones > zeros:
        raise Exception("Check your scaling, most of the image is 1's")
    return x

###IMPORT AND IDENTIFY IMAGES###

def latexCommand(index):
    line=index+1
    file = open("C:/Users/Joe/Documents/S17/15-112/Term Project/HASYv2/symbols3.csv",'r')
    reader = csv.reader(file)
    for i,row in enumerate(reader):
        if i==line:
            return row[1]
    print ("you fucked up")

def formatURL(input,data):
    inp = str(input)
    inp = inp.strip()
    inp = inp.replace("+","%2B")
    inp = inp.replace(" ","+")
    inp+="%3F"
    inp+="&width="
    inp+=str(data.width//2-data.margin)
    return inp

#Wolfrom Alpha API Key obtained from wolframalpha.com
#Used the Wolfram Alpha module which can be pip installed via 'wolframalpha'
import urllib

def wolframAlpha(input,data):
    appID = "4YUQ4H-EUKJ63VXG2"
    url = 'http://api.wolframalpha.com/v1/simple?appid=4YUQ4H-EUKJ63VXG2&i='
    query = formatURL(input,data)
    url+=query
    #Following syntax loosely taken from
    # "http://stackoverflow.com/questions/40911170/ \n
    # python-how-to-read-an-image-from-a-url
    try:
        image = Image.open(urllib.request.urlopen(url))
        image.save("wolframTemp.gif","gif")
        return 1 #to differentiate between returning None
    except:
        print ("Connect to the internet or enter a valid query")
        return None

###GRAPHICS AND GUI###
# mouseEventsDemo.py
# TAKEN AND MODIFIED FROM 15-112 WEBSITE #

from tkinter import (Tk,ALL,PhotoImage,Canvas,simpledialog,messagebox,
                     Frame,Label,Entry,NW,CENTER)
###BUTTONS AND WINDOWS###
class Window1(simpledialog.Dialog): #taken from the 15-112 website
    def body(self, master):
        self.modalResult = None
        Label(master, text="Correct Symbol \n   \
        (press OK without entering anything \n  \
        if the digit is already correct):").grid(row=0)
        self.e1 = Entry(master)
        self.e1.grid(row=0, column=1)
        return self.e1  # initial focus

    def apply(self):
        first = self.e1.get()
        self.modalResult = (first)

def showDialog(data): #taken from the 15-112 website
    dialog = Window1(data.root)
    return dialog.modalResult

class Button(object):
    def __init__(self,x0,y0,x,y,fgcolor,bgcolor,data,text,textcolor="black"):
        self.x = x0
        self.y = y0
        self.dims=(x0-x//2,y0-y//2,x0+x//2,y0+y//2)
        self.normalColor=fgcolor
        self.clickedColor=bgcolor
        self.textColor=textcolor
        self.text=text
        self.data = data
    def drawNormal(self,canvas):
        canvas.create_rectangle(self.dims,fill=self.normalColor)
        canvas.create_text(self.x,self.y,text = self.text,fill=self.textColor,
                           font=("Bradley Hand ITC",14,"bold"),justify=CENTER)
    def drawClicked(self,canvas):
        canvas.create_rectangle(self.dims,fill=self.clickedColor)
        canvas.create_text(self.x, self.y, text=self.text, fill=self.textColor)
    def inBoundaries(self,x,y):
        x0,y0,x1,y1=self.dims
        if x<=x1 and x>=x0 and y<=y1 and y>=y0:
            return True
        return False
    def drawImage(self,canvas):
        image = self.data.eraserImage
        canvas.create_image(self.x,self.y,image=image)

###MODEL###
def startButton(data):
    data.splash = False
    data.draw = True
    data.erase = False

def classifyButton(data):
    classifySymbols(data)
    convertClassification(data)

def clearButton(data):
    data.symbol = set()
    data.classification = []
    data.characters = []

def correctButton(data):
    train_x,train_y=[],[]
    findBoundaries(data)
    for (symbol,i) in zip(data.characters,list(range(len(data.characters)))):
        (left, top, right, bottom) = symbol
        mnist = resizeImagetoSquare(left, top, right, bottom, data)[0]
        hasy = resizeImagetoSquare(left, top, right, bottom, data)[1]
        classification = data.classification[i]
        data.highlight = classification
        try:
            correct = str(showDialog(data))
            data.highlight = ""
            if correct == "" or None:
                continue
            data.classification[i] = correct
            onlineLearning(data,hasy,mnist,correct)
        except:
            continue
        convertClassification(data)

def solveButton(data):
    if wolframAlpha(data.printable,data) == None:
        return None
    wolframAlpha(data.printable,data)
    showWolframAlpha(data)
    data.solved = True
    data.draw = False

def backButton(data):
    data.splash=True
    data.about=False
    data.draw = False
    data.symbol = set()
    data.classification = []
    data.characters = []
    data.erase = False

def aboutButton(data):
    data.about=True
    data.splash=False

def solveBackButton(data):
    clearButton(data)
    data.solved = False
    data.draw = True

def eraseButton(data):
    data.erase = not(data.erase)
    if data.erase == False:
        data.eraserImage = PhotoImage(file="eraser.gif")
    else:
        data.eraserImage = PhotoImage(file="chalk.gif")

def make2dList(rows,cols): #makes 2d list- similar to 15-112 website
    return [[0 for i in range(cols)] for i in range(rows)]

def findBoundaries(data):
    #positions of every symbol drawn
    rows = data.height//data.squaresize
    cols = data.width//data.squaresize
    data.image = make2dList(rows,cols)
    imageList = data.image
    #create 2D list of 1's and 0's modeling the entire image
    for position in data.symbol:
        row,col = position[1],position[0]
        imageList[row][col] = 1

    left,right,top,bottom = -1,-1,-1,-1
    intermediate = -2 #helps determine where the left edge is in the middle
    #of the page

    # find the top and bottom-most shits
    for row in range(rows):
        for col in range(cols):
            if imageList[row][col]==1:
                if top==-1:
                    top = row #saves highest row
                bottom = row #saves lowest row
    # find left-most boundary (lowest x-val):
    for col in range(len(imageList[0])):
        count = 0
        for row in range(rows):
            if imageList[row][col]==1:
                count+=1
                #saves the right-most column in which there is a black pixel
                right = col
                if left==-1 or left==intermediate:
                    #saves the left-most column in which there is a black pixel
                    left = col
        if count==0 and right>left:
            boundaries = (left, top, right + 1, bottom + 1)
            if boundaries not in data.characters:
                data.characters.append(boundaries)
            intermediate = left

    return data.characters

def resizeImagetoSquare(left,top,right,bottom,data):
    image = make2dList((bottom-top),(right-left))
    for i in range(left,right):
        for j in range(top,bottom):
            image[j-top][i-left] = data.image[j][i]
    #Remove white space at top and bottom
    for row in image:
        if 1 not in row:
            image.remove(row)

    #convert to image
    image = Image.fromarray(np.array(image))
    hasyimage = image
    ##For MNIST##
    imageCenter = image.resize((20,20))
    imarr = np.array(imageCenter)
    mnistImage = np.zeros((28,28))
    mnistImage[4:24,4:24] = imarr
    ##For HASY##
    hasyimage = hasyimage.resize((32,32))
    hasyimage = np.array(hasyimage)
    return np.reshape(mnistImage,(784,1)),np.reshape(hasyimage,(1024,1))

def findSquare(x,y,data):
    squaresize = data.squaresize
    i = x//squaresize
    j = y//squaresize
    if (i,j) not in data.symbol:
        data.symbol.add((i,j))
    if (i+1,j) not in data.symbol:
        data.symbol.add((i+1,j))
    if (i,j+1) not in data.symbol:
        data.symbol.add((i,j+1))
    if (i+1,j+1) not in data.symbol:
        data.symbol.add((i+1,j+1))

def findErase(x,y,data):
    squaresize = data.squaresize
    i = x // squaresize
    j = y // squaresize
    if (i, j) in data.symbol:
        data.symbol.remove((i, j))
    if (i + 1, j) in data.symbol:
        data.symbol.remove((i + 1, j))
    if (i, j + 1) in data.symbol:
        data.symbol.remove((i, j + 1))
    if (i + 1, j + 1) in data.symbol:
        data.symbol.remove((i + 1, j + 1))

def init(data):
    #Misc.
    data.margin = 20
    data.printable=""
    data.buttonColor = "grey"
    data.splash = True
    data.about = False
    data.solved = False
    data.draw = False
    data.erase = False
    data.symbol = set()
    data.size = 110
    data.squaresize = data.height // data.size
    data.image = []
    data.classification = []
    data.highlight = ""
    data.net = createNet()
    data.net2 = createMNIST()
    data.characters = []  # list containing tuples of the (left,top,right,bottom)
    #Images
    data.pic = PhotoImage(file="background.gif") #taken from www.123rf.com
    data.drawpic= PhotoImage(file="blackboard.gif") #taken from www.123RF.com
    data.eraserImage = PhotoImage(file="eraser.gif") #taken from www.123rf.com
    data.aboutpic = PhotoImage(file="about_screen.gif") #created on my own
    #Buttons
    data.startButton = Button(data.width//3,3*data.height//4,90,50,
                              data.buttonColor,"white",data,"Start")
    data.classifyButton = Button(data.width//3-5,5*data.height//6,90,50,
                                 data.buttonColor,"white",data,"Classify")
    data.clearButton = Button(5+2*data.width//3,5*data.height//6,90,50,
                              data.buttonColor,"white",data,"Clear")
    data.correctButton = Button(data.width//3-5,5*data.height//6,90,50,
                                data.buttonColor,"white",data,"Correct It")
    data.solveButton = Button(data.width//2,5*data.height//6,140,50,
                data.buttonColor,"white",data,"Send To \n Wolfram Alpha!")
    data.backButton = Button(45,20,90,40,data.buttonColor,"white",data,"Back")
    data.aboutButton = Button(2*data.width//3,3*data.height//4,90,50,
                              data.buttonColor,"white",data,"About")
    data.solveBackButton = Button(45,20,90,40,data.buttonColor,'white',
                                  data,"Back")
    data.eraseButton = Button(data.width-45,20,90,40,data.buttonColor,"white",
                              data,"shouln't ever be displayed")

def classifySymbols(data):
    findBoundaries(data)
    for symbol in data.characters:
        (left, top, right, bottom) = symbol
        mnist = resizeImagetoSquare(left, top, right, bottom, data)[0]
        hasy = resizeImagetoSquare(left, top, right, bottom, data)[1]
        hasy2 = data.net.feedforward(hasy)
        mnist2 = data.net2.feedforward(mnist)
        exclusions = [0,78, 79, 80,87]
        for i in exclusions:
            hasy2[i][0] = 0
        has = np.amax(hasy2)
        mnst = (np.amax(mnist2))*1.05
        if has >= mnst:
            line = np.argmax(hasy2)
            data.classification.append(str(latexCommand(line)))

        else:
            index = np.argmax(mnist2)
            data.classification.append(str(index))

def onlineLearning(data,hasyarray,mnistarray,inp):
    inputy = str(inp)
    if inputy in string.digits:
        net = data.net2
        array = mnistarray
        index = int(inp)
        title = 'JUSTDIDTHISTONIGHT'
    else:
        net = data.net
        array = hasyarray
        index = findIndex(inp)
        title = 'hasy'
    testY = make2dList(net.netSize[2],1)
    testY[int(index)][0] = 1
    net.MBGD([array],[testY])
    net.save(title)

def findIndex(input):
    file = open(
        "C:/Users/Joe/Documents/S17/15-112/Term Project/HASYv2/symbols3.csv", 'r')
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if str(input).lower()==str(row[1]).lower():
            return i-1
    print ("Symbol Name Not Recognized")
    return None

def convertClassification(data):
    classification = ""
    for i in data.classification:
        classification+=i+' '
    exclusions = ["f",'m','s','l','+','-','*','/']
    for index in range(1,len(classification)-1):
        try:
            if ((classification[index - 1] in string.ascii_letters) and \
                        (classification[index + 1] in string.digits) and \
                        (classification[index-1] not in exclusions) and \
                                    classification[index] == " "):
                classification=classification[:index]+"^"+classification[index+1:]
        except:
            continue
    for index in range(1,len(classification)-1):
        try:
            if ((classification[index-1] in string.digits) and \
                (classification[index+1] in string.digits) and \
                                    classification[index]==" "):
                classification = classification[:index]+classification[index+1:]
        except:
            continue
    data.printable=classification

def showWolframAlpha(data):
    im = Image.open("wolframtemp.gif")
    (width,height) = im.size
    h = data.height-data.margin
    im1 = im.crop(box=(0,0,width,h))
    im1.save("temp1.gif")
    try:
        im2 = im.crop(box=(0,h,width,height))
    except:
        im2 = im.crop(box=(0,h,width,2*h))
    im2.save("temp2.gif")
    data.im1 = PhotoImage(file="temp1.gif")
    data.im2 = PhotoImage(file="temp2.gif")
###CONTROLLER###
def leftReleased(event, data):
    setEventInfo(event, data, "leftReleased")
    data.leftPosn = (event.x, event.y)

def setEventInfo(event, data, eventName):
    ctrl  = ((event.state & 0x0004) != 0)
    shift = ((event.state & 0x0001) != 0)
    msg = ""
    if ctrl:  msg += "ctrl-"
    if shift: msg += "shift-"
    msg += eventName
    msg += " at " + str((event.x, event.y))
    data.info = msg

def mouseMotion(event,data):
    setEventInfo(event, data, "mouseMotion")
    data.motionPosn = (event.x, event.y)

def leftPressed(event, data):
    setEventInfo(event, data, "leftPressed")
    data.leftPosn = (event.x, event.y)
    if data.splash==True:
        if data.startButton.inBoundaries(event.x,event.y):
            startButton(data)
        if data.aboutButton.inBoundaries(event.x,event.y):
            aboutButton(data)
    elif data.about==True:
        if data.backButton.inBoundaries(event.x, event.y):
            backButton(data)
    elif data.draw==True:
        if data.classification != []:  #Already Classified
            if data.correctButton.inBoundaries(event.x, event.y):
                correctButton(data)
            elif data.solveButton.inBoundaries(event.x,event.y):
                solveButton(data)
            elif data.backButton.inBoundaries(event.x, event.y):
                backButton(data)
            elif data.clearButton.inBoundaries(event.x, event.y):
                clearButton(data)
        else:  #still drawing
            if data.classifyButton.inBoundaries(event.x,event.y):
                classifyButton(data)
            elif data.backButton.inBoundaries(event.x, event.y):
                backButton(data)
            elif data.clearButton.inBoundaries(event.x, event.y):
                clearButton(data)
            elif data.eraseButton.inBoundaries(event.x,event.y):
                eraseButton(data)
    elif data.solved==True:
        if data.solveBackButton.inBoundaries(event.x,event.y):
            solveBackButton(data)

def leftMoved(event, data):
    setEventInfo(event, data, "leftMoved")
    data.leftPosn = (event.x, event.y)
    if data.splash==False and data.erase==False:
        findSquare(event.x, event.y, data)
    elif data.erase==True:
        findErase(event.x,event.y,data)

def timerFired(data): pass

def keyPressed(event, data): pass

###VIEW###
def createGrid(canvas,data):
    width = data.width
    height = data.height
    squaresize = data.squaresize
    for square in data.symbol:
        i,j = square[0],square[1]
        x0,y0,x1,y1=i*squaresize,j*squaresize,(i+1)*squaresize,(j+1)*squaresize
        canvas.create_rectangle(x0,y0,x1,y1,fill="white",outline = "white")

def drawSplashScreen(canvas,data):
    if data.splash == True:
        #background image
        canvas.create_image(data.width//2,data.height//2, image=data.pic)
        #above code (background image) taken from 15-112 website
        data.startButton.drawNormal(canvas)
        data.aboutButton.drawNormal(canvas)
    elif data.about==True:
        canvas.create_image(data.width//2,data.height//2,image=data.drawpic)
        im = data.aboutpic
        canvas.create_image(data.width//2,data.height//2+15,image=im)

def drawClassifyScreen(canvas,data):
    if data.draw==True: #Draw Screen
        if data.classification != []: #already classified
            canvas.create_image(data.width//2,data.height//2,image=data.drawpic)
            canvas.create_text(data.width//2,data.height//2,text=str(data.printable),
                            font=("Bradley Hand ITC", 20,"bold"),fill="white")
            data.correctButton.drawNormal(canvas)
            data.solveButton.drawNormal(canvas)
            if data.highlight != "":
                text = "Classified Symbol: "+str(data.highlight)
                canvas.create_text(data.width//2,2*data.height//3,
                text=text,font=("Bradley Hand ITC", 20,"bold"),fill="white")
        else:
            canvas.create_image(data.width//2,data.height//2,image=data.drawpic)
            createGrid(canvas,data)
            drawSquareBoundaries(canvas,data)
            canvas.create_text(data.width//2,data.height//10,text="Write Math",
                               font=("Bradley Hand ITC",20,"bold"),fill="white")
            data.classifyButton.drawNormal(canvas)
            data.eraseButton.drawImage(canvas)
        data.clearButton.drawNormal(canvas)
        data.backButton.drawNormal(canvas)
    elif data.splash==False: #About Screen
        data.backButton.drawNormal(canvas)

def drawSolvedScreen(canvas,data):
    if data.solved == True:
        im1 = data.im1
        im2 = data.im2
        canvas.create_rectangle(0,0,data.width,data.height,fill="white")
        canvas.create_image(20,20,anchor=NW,image=im1)
        canvas.create_image(data.width//2,20,anchor=NW,image=im2)
        data.solveBackButton.drawNormal(canvas)

def drawSquareBoundaries(canvas,data):
    for positions in data.characters:
        (x0,y0,x1,y1) = positions
        s = data.squaresize
        canvas.create_rectangle(x0*s,y0*s,x1*s,y1*s,fill="",outline="black")

def redrawAll(canvas, data):
    drawSplashScreen(canvas,data)
    drawClassifyScreen(canvas,data)
    drawSolvedScreen(canvas,data)

####################################
# use the run function as-is
####################################

def run(width=800, height=400):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    # Note changes #1:
    def mouseWrapper(mouseFn, event, canvas, data):
        mouseFn(event, data)
        #redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 20 # milliseconds
    root = Tk()
    data.root = root
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.grid()
    # set up events

    # Note changes #2:
    root.bind("<Button-1>", lambda event:
                            mouseWrapper(leftPressed, event, canvas, data))
    #root.bind("<Button-3>", lambda event:
                            #mouseWrapper(rightPressed, event, canvas, data))
    canvas.bind("<Motion>", lambda event:
                            mouseWrapper(mouseMotion, event, canvas, data))
    canvas.bind("<B1-Motion>", lambda event:
                            mouseWrapper(leftMoved, event, canvas, data))
    #canvas.bind("<B3-Motion>", lambda event:
                            #mouseWrapper(rightMoved, event, canvas, data))
    root.bind("<B1-ButtonRelease>", lambda event:
                            mouseWrapper(leftReleased, event, canvas, data))
    #root.bind("<B3-ButtonRelease>", lambda event:
                            #mouseWrapper(rightReleased, event, canvas, data))

    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

#Overview of Citations#
#Backpropogation Algorithm-- within my Network Class there is a method called
#backprop(x,y) which was taken entirely and without modification from Michael
#Neilsen's book Neural Networks and Deep Learning. This can be found online at
#http://neuralnetworksanddeeplearning.com/chap1.html

#Several of my functions for loading data call external python files called
#'mnist_training' and 'unpackData'.  These files are a mixture of my own code
#and code written by others.  See the files for more detailed citations

#I use the wolframAlpha module which is a nice way of accessing the Wolfram API
#more information about the module and API can be found at www.wolframalpha.com

#My pop-up dialog class was taken with light modifications from the 15-112 website
#under miscelaneous tkinter demos

#The entirety of my graphics is built of events_example0.py from the 15-112 website

#the run function is modified from mouse-pressed examples posted on the 15-112 website

#any media (pictures) used in this project were are clearly cited in a comment
#to the right of where they are first called. (generally in init)

#In a couple locations I load or save files using code taken and modified from
#stackoverflow.com.  The exact URL's of these can be found next to the usage.

#All above citations can be found next to their location in my code#