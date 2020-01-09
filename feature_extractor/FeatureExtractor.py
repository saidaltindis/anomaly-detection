import os
import shelve
import RGB_OF_API as ROA
import NN_FeatureExctractor_API as NNF

DATASET_PATH = '/home/saidaltindis/ALL FILES/PROJECTS/1SENIOR/data/Resized Data/'
OUTPUT_PATH = '/home/saidaltindis/ALL FILES/PROJECTS/1SENIOR/data/Neural Network/'
OUTPUT_FILE_NAME = 'NeuralNetworkFeatures.shelve'
TRAIN_TEST_SPLIT_PATH = '/home/saidaltindis/ALL FILES/PROJECTS/1SENIOR/eyes-on-you/train_test_split/'


def extractFeatures(categoryName):
    
    train, test = readTrainTestSplit(categoryName)
    videoFolderPath = DATASET_PATH + categoryName + '/'
    videoCounter = 0
    with os.scandir(videoFolderPath) as videos:
         for video in videos:
            print("[INFO] --> Number of video: '%s'" % str(videoCounter))
            isExist = checkIfExists(video.name, categoryName)
            if isExist == 0:
                rgb = ROA.computeRGB(video)
                of = ROA.computeOF(video)
                rgbFeature, opticalFlowFeature = NNF.extractNNFeatures(rgb, of)

                if video.name in train:
                    saveFeatures(rgbFeature, opticalFlowFeature, video.name, categoryName, "train")
                else:
                    saveFeatures(rgbFeature, opticalFlowFeature, video.name, categoryName, "test")
            else:
                print("[INFO] --> '%s' already exists. Skipping." % video.name)
            videoCounter = videoCounter + 1

def checkIfExists(videoName, categoryName):
    print("[INFO] --> Checking if feature of '%s' already exists" % videoName)

    OUTPUT_FILE = OUTPUT_PATH + categoryName + '/' + OUTPUT_FILE_NAME
    featureDB = shelve.open(OUTPUT_FILE)

    flag = 0

    try:
        categoryFeature = featureDB[categoryName]
        trainSet = categoryFeature.get('train')
        testSet = categoryFeature.get('test')
        for data in trainSet:
            if videoName == data.get('name'):
                flag = 1
        for data in testSet:
            if videoName == data.get('name'):
                flag = 1
    except KeyError:
        flag = 0
    finally:
        featureDB.close()

    return flag

def saveFeatures(rgbFeature, opticalFlowFeature, videoName, categoryName, type):
    print('[INFO] --> Saving model features for "%s" to shelve.' % videoName)
    
    data = {
        'name': videoName,
        'features': {
            'rgb': rgbFeature,
            'flow': opticalFlowFeature
        }
    }

    OUTPUT_FILE = OUTPUT_PATH + categoryName + '/' + OUTPUT_FILE_NAME
    featureDB = shelve.open(OUTPUT_FILE)
    
    try:
        categoryFeature = featureDB[categoryName]
        categoryFeature.get(type).append(data)
        featureDB[categoryName] = categoryFeature
    except KeyError:
        categoryFeature = {'train': [], 'test': []}
        categoryFeature.get(type).append(data)
        featureDB[categoryName] = categoryFeature
    finally:
        featureDB.close()

def readFeatures(categoryName):
    print("[INFO] --> Reading model features from shelve to dictionary.")
    OUTPUT_FILE = OUTPUT_PATH + categoryName + '/' + OUTPUT_FILE_NAME
    featureDB = shelve.open(OUTPUT_FILE)
    features = featureDB[categoryName];
    print(features)

    return features;

def readTrainTestSplit(categoryName):
    
    train = []
    test = []

    directoryPath = TRAIN_TEST_SPLIT_PATH + categoryName + '/'

    trainPath = directoryPath + 'train.txt'
    testPath = directoryPath + 'test.txt'

    trainFile = open(trainPath, 'r')
    trainLines = trainFile.readlines();
    trainFile.close()

    testFile = open(testPath, 'r')
    testLines = testFile.readlines();
    testFile.close()

    for line in trainLines:

        line = line.rstrip('\n')
        train.append(line)
    
    for line in testLines:
        line = line.rstrip('\n')
        test.append(line)
    
    return train, test



extractFeatures('Fighting')
