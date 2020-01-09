import os
import shelve

INPUT_PATH = '/home/saidaltindis/ALL FILES/PROJECTS/1SENIOR/data/Neural Network/'
INPUT_FILE_NAME = 'NeuralNetworkFeatures.shelve'
OUTPUT_PATH = '/home/saidaltindis/ALL FILES/PROJECTS/1SENIOR/data/ML Inputs/'
OUTPUT_FILE_NAME = 'MLInputs.shelve'

def inputOrgz(categoryName):

    data = {
        'rgb': {
            'train_X': [], 'train_Y': [], 'test_X': [], 'test_Y': []
        },
        'flow':{
            'train_X': [], 'train_Y': [], 'test_X': [], 'test_Y': []
        }
    }

    indexer = 0
    indexDict = {'index': [], 'name': []}

    features = readFeatures(categoryName)
    trainDict = features.get('train')
    testDict = features.get('test')

    categoryNo = defineCategoryNo(categoryName)
    
    for sample in trainDict:
        indexDict.get('index').append(indexer); indexer = indexer + 1;
        indexDict.get('name').append(sample.get('name'))

        data.get('rgb').get('train_X').append(sample.get('features').get('rgb'))
        data.get('rgb').get('train_Y').append(categoryNo)  
        data.get('flow').get('train_X').append(sample.get('features').get('flow'))
        data.get('flow').get('train_Y').append(categoryNo)
    
    indexer  = 0
    for sample in testDict:
            indexDict.get('index').append(indexer); indexer = indexer + 1;
            indexDict.get('name').append(sample.get('name'))

            data.get('rgb').get('test_X').append(sample.get('features').get('rgb'))
            data.get('rgb').get('test_Y').append(categoryNo)  
            data.get('flow').get('test_X').append(sample.get('features').get('flow'))
            data.get('flow').get('test_Y').append(categoryNo)

    saveMLInputs(data, indexDict, categoryName)

def saveMLInputs(data, index, categoryName):
    print("[INFO] --> Saving ML inputs to shelve to dictionary.")
    SHELVE_FILE = OUTPUT_PATH + categoryName + OUTPUT_FILE_NAME
    
    fileToSave = {'inputs': data, 'indexes': index}

    inputDB = shelve.open(SHELVE_FILE)
    inputDB[categoryName] = fileToSave
    print(fileToSave)

def readFeatures(categoryName):
    print("[INFO] --> Reading model features from shelve to dictionary.")
    SHELVE_FILE = INPUT_PATH + categoryName + '/' + INPUT_FILE_NAME
    featureDB = shelve.open(SHELVE_FILE)
    features = featureDB[categoryName]
    return features

def defineCategoryNo(categoryName):
    categoryNo = 0
    if categoryName == 'Fighting':
        categoryNo = 1
    elif categoryName == 'RoadAccidents':
        categoryNo = 2
    else:
        categoryNo = 3
    return categoryNo


inputOrgz("RoadAccidents")
