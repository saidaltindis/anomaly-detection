import os
import re

ANNOTATION_PATH = '../annotations/'
DATASET_PATH = '/home/saidaltindis/ALL FILES/DATASETS/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Videos/'
OUTPUT_PATH = '/home/saidaltindis/ALL FILES/PROJECTS/1SENIOR/data/Cropped Data/'
ANNOTATION_FILE_PREFIX = '.txt'

def read_annotation_file(categoryName):
    
    annotationInfo = {
        'filename': [],
        'startTime': [],
        'endTime': [],
        'duration': []
    }

    annotationFilePath = ANNOTATION_PATH + categoryName + ANNOTATION_FILE_PREFIX

    annotationFile = open(annotationFilePath, 'r')
    lines = annotationFile.readlines();

    for line in lines:

        line = line.rstrip('\n')
        elems = line.split('    ');

        annotationInfo.get('filename').append(elems[0])
        annotationInfo.get('startTime').append(elems[1])
        annotationInfo.get('endTime').append(elems[2])
        annotationInfo.get('duration').append(elems[3])
    
    return annotationInfo


def crop_videos(categoryName, annotationInfo):
    
    videoPath = DATASET_PATH + categoryName + '/'
    os.mkdir(OUTPUT_PATH + categoryName)

    filenames = annotationInfo.get('filename')
    startTimes = annotationInfo.get('startTime')
    endTimes = annotationInfo.get('endTime')
    
    with os.scandir(videoPath) as videos:    
        for video in videos:
            if video.name in filenames:
                index = filenames.index(video.name)
                inputFile = videoPath + video.name
                outputFile = OUTPUT_PATH + categoryName + '/' + video.name
                
                commandToExecute = 'ffmpeg -i \'' + inputFile + '\' -ss ' + startTimes[index] + ' -to ' + endTimes[index] + ' -async 1 \'' + outputFile + '\''
                os.system(commandToExecute)
            else:
                continue