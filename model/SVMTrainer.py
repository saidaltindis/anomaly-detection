import shelve
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

'''
    BEFORE RUNNING SVM TRAINER, DO FOLLOWING
    1. Crop and resize videos of category.
    2. Extract RGB and optical flow features to eventually draw neural network features.
    3. Use InputOrganizer to create appropriate input for SVM.
    4. Finally, you can run SVMTrainer.
'''

INPUT_PATH = '/home/saidaltindis/ALL FILES/PROJECTS/1SENIOR/data/ML Inputs/'
INPUT_FILE_NAME = 'MLInputs.shelve'
OUTPUT_PATH = '/home/saidaltindis/ALL FILES/PROJECTS/1SENIOR/data/SVM Models/'


CATEGORIES_TO_TRAIN = ["Normal", "RoadAccidents"]

def run(CATEGORIES_TO_TRAIN):
    rgb, flow = concatCategories(CATEGORIES_TO_TRAIN)
    print("[INFO] --> Training with RGB features")
    rgbp, classes = SVMLearning(rgb, 'rgb')
    print("[INFO] --> Training with Optical Flow features")
    flowp, classes = SVMLearning(flow, 'flow')
    print("[INFO] --> Fusing results of RGB and Optical Flow Learners")
    fusedResults = fuseRGBandFlowResults(rgbp, flowp, classes)
    print("[INFO] --> Results")
    successOfResults(fusedResults, rgb.get('test_Y'))

def runWithLoad(CATEGORIES_TO_TRAIN):
    rgb, flow = concatCategories(CATEGORIES_TO_TRAIN)
    print("[INFO] --> Training with RGB features")
    rgbm, classes = loadModel(OUTPUT_PATH, 'rgb')
    print("[INFO] --> Training with Optical Flow features")
    flowm, classes = loadModel(OUTPUT_PATH, 'flow')
    print("[INFO] --> Fusing results of RGB and Optical Flow Learners")
    fusedResults = fuseRGBandFlowResults(rgbm.predict_proba(rgb.get('test_X')), flowm.predict_proba(flow.get('test_X')), classes)
    print("[INFO] --> Results")
    successOfResults(fusedResults, rgb.get('test_Y'))   

def SVMLearning(data, modelName):
    SVMLearner = SVC(kernel = 'linear', probability=True)
    SVMLearner.fit(data.get('train_X'), data.get('train_Y'))
    #saveModel(SVMLearner, modelName)
    results = SVMLearner.predict(data.get('test_X'))
    return SVMLearner.predict_proba(data.get('test_X')), SVMLearner.classes_

def fuseRGBandFlowResults(rgbResultsProb, flowResultsProb, classes):
    
    fuseResults = []
    numberOfSamples = len(rgbResultsProb)

    for i in range(0, numberOfSamples):
        weightedResult = (0.5)*rgbResultsProb[i][0] + (0.5)*flowResultsProb[i][0]
        if weightedResult > 0.5:
            fuseResults.append(classes[0])
        else:
            fuseResults.append(classes[1])
    
    return fuseResults

def successOfResults(predictedResults, realResults):
    confMatrix = confusion_matrix(realResults, predictedResults)
    tpScore = ((confMatrix[1,1])/(confMatrix[1,1]+confMatrix[1,0]))
    score = ((confMatrix[0,0]+confMatrix[1,1])/(confMatrix[0,0]+confMatrix[1,1]+confMatrix[0,1]+confMatrix[1,0]))

    print('--------------')
    print(score)
    print('--------------')
    print(tpScore)
    print('--------------')
    print(confMatrix)
    print('--------------')

def saveModel(model, modelName):
    OUTPUT_FILE_NAME = OUTPUT_PATH + modelName
    file = open(OUTPUT_FILE_NAME, 'wb')
    pickle.dump(model, file)
    file.close()

def loadModel(pathToModel, modelName):
    MODEL_FILE_NAME = pathToModel + modelName
    file = open(MODEL_FILE_NAME, 'rb')
    model = pickle.load(file)
    file.close()
    return model, model.classes_

def concatCategories(categories):
    print("[INFO] --> Concatenating ML Inputs of" + categories[0] + " and " + categories[1])
    
    NORMAL_VIDEO_TRAIN_LIMIT, NORMAL_VIDEO_TEST_LIMIT, anomalyCategory = normalVideoLimitDefiner(categories)
    indexOfNormal = categories.index('Normal')
    indexOfAnomaly = categories.index(anomalyCategory)

    categoryInputs = []
    for category in categories:
        categoryInputs.append(readFeatures(category))
    
    rgb = {'train_X': [], 'train_Y': [], 'test_X': [], 'test_Y': []}
    flow = {'train_X': [], 'train_Y': [], 'test_X': [], 'test_Y': []}

    anomalyInputs = categoryInputs[indexOfAnomaly].get('inputs')
    normalInputs = categoryInputs[indexOfNormal].get('inputs')
    # Adding anomaly to train_X
    rgb['train_X'] = anomalyInputs.get('rgb').get('train_X')
    flow['train_X'] = anomalyInputs.get('flow').get('train_X')
    # Adding normal to train_X
    rgb['train_X'] = rgb.get('train_X') + normalInputs.get('rgb').get('train_X')[0:NORMAL_VIDEO_TRAIN_LIMIT]
    flow['train_X'] = flow.get('train_X') + normalInputs.get('flow').get('train_X')[0:NORMAL_VIDEO_TRAIN_LIMIT]
    # Adding anomaly to train_Y
    rgb['train_Y'] = anomalyInputs.get('rgb').get('train_Y')
    flow['train_Y'] = anomalyInputs.get('flow').get('train_Y')
    # Adding normal to train_Y
    rgb['train_Y'] = rgb.get('train_Y') + normalInputs.get('rgb').get('train_Y')[0:NORMAL_VIDEO_TRAIN_LIMIT]
    flow['train_Y'] = flow.get('train_Y') + normalInputs.get('flow').get('train_Y')[0:NORMAL_VIDEO_TRAIN_LIMIT]
    # Adding anomaly to test_X
    rgb['test_X'] = anomalyInputs.get('rgb').get('test_X')
    flow['test_X'] = anomalyInputs.get('flow').get('test_X')   
    # Adding normal to test_X
    rgb['test_X'] = rgb.get('test_X') + normalInputs.get('rgb').get('test_X')[0:NORMAL_VIDEO_TEST_LIMIT]
    flow['test_X'] = flow.get('test_X') + normalInputs.get('flow').get('test_X')[0:NORMAL_VIDEO_TEST_LIMIT]   
    # Adding anomaly to test_Y
    rgb['test_Y'] = anomalyInputs.get('rgb').get('test_Y')
    flow['test_Y'] = anomalyInputs.get('flow').get('test_Y')   
    # Adding normal to test_Y
    rgb['test_Y'] = rgb.get('test_Y') + normalInputs.get('rgb').get('test_Y')[0:NORMAL_VIDEO_TEST_LIMIT]
    flow['test_Y'] = flow.get('test_Y') + normalInputs.get('flow').get('test_Y')[0:NORMAL_VIDEO_TEST_LIMIT]

    # Converting all to the numpy array
    rgb['train_X'] = np.asarray(rgb.get('train_X'))
    rgb['train_Y'] = np.asarray(rgb.get('train_Y'))
    rgb['test_X'] = np.asarray(rgb.get('test_X'))
    rgb['test_Y'] = np.asarray(rgb.get('test_Y'))
    flow['train_X'] = np.asarray(flow.get('train_X'))
    flow['train_Y'] = np.asarray(flow.get('train_Y'))
    flow['test_X'] = np.asarray(flow.get('test_X'))
    flow['test_Y'] = np.asarray(flow.get('test_Y'))

    # Reshape all
    rgb['train_X'] = np.reshape(rgb.get('train_X'), (rgb.get('train_X').shape[0], rgb.get('train_X').shape[2]))
    rgb['test_X'] = np.reshape(rgb.get('test_X'), (rgb.get('test_X').shape[0], rgb.get('test_X').shape[2]))
    flow['train_X'] = np.reshape(flow.get('train_X'), (flow.get('train_X').shape[0], flow.get('train_X').shape[2]))
    flow['test_X'] = np.reshape(flow.get('test_X'), (flow.get('test_X').shape[0], flow.get('test_X').shape[2]))

    return rgb, flow

def readFeatures(categoryName):
    print("[INFO] --> Reading ML Inputs from shelve to dictionary.")
    SHELVE_FILE = INPUT_PATH + categoryName + INPUT_FILE_NAME
    featureDB = shelve.open(SHELVE_FILE)
    features = featureDB[categoryName]
    return features

def normalVideoLimitDefiner(categories):
    trainLimit = 0
    testLimit = 0
    anomalyCategory = ""
    if "Fighting" in categories:
        print("[INFO] --> Setting normal video limits according to Fighting")
        anomalyCategory = "Fighting"
        trainLimit = 35
        testLimit = 15
    else:
        print("[INFO] --> Setting normal video limits according to Road Accident")
        anomalyCategory = "RoadAccidents"
        trainLimit = 100
        testLimit = 50
    
    return trainLimit, testLimit, anomalyCategory
    
#run(CATEGORIES_TO_TRAIN)
runWithLoad(CATEGORIES_TO_TRAIN)
