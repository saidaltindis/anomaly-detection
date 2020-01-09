import os

VIDEO_HEIGHT = '224'
VIDEO_WIDTH = '224'

CROPPED_VIDEO_PATH = '/home/saidaltindis/ALL FILES/PROJECTS/1SENIOR/data/Cropped Data/'
OUTPUT_VIDEO_PATH = '/home/saidaltindis/ALL FILES/PROJECTS/1SENIOR/data/Resized Data/'

def resize_videos(categoryName):
    
    videoPath = CROPPED_VIDEO_PATH + categoryName + '/'
    os.mkdir(OUTPUT_VIDEO_PATH + categoryName)

    with os.scandir(videoPath) as videos:    
        for video in videos:
            inputFile = videoPath + video.name
            outputFile = OUTPUT_VIDEO_PATH + categoryName + '/' + video.name
                
            commandToExecute = 'ffmpeg -i \'' + inputFile + '\' -vf scale=' + VIDEO_HEIGHT + ':' + VIDEO_WIDTH + ' \'' + outputFile + '\''
            os.system(commandToExecute)