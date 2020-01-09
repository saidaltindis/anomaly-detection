import os
import sys
import cv2  
import numpy as np
import time

VIDEO_FILE_PREFIX = '.mp4'

def computeRGB(video):
    print('[INFO] --> Computing RGB for the video: "%s"' % video.name)
    rgb = []
    vidcap = cv2.VideoCapture(video.path)
    success,frame = vidcap.read()
    while success:
        frame = cv2.resize(frame, (342,256)) 
        frame = (frame/255.)*2 - 1
        frame = frame[16:240, 59:283]    
        rgb.append(frame)        
        success,frame = vidcap.read()
    vidcap.release()
    rgb = rgb[:-1]
    rgb = np.asarray([np.array(rgb)])
    print('[INFO] --> Shape of the computed rgb: ',rgb.shape)
    #np.save(SAVE_DIR+'/rgb.npy', rgb)
    return rgb

def computeOF(video):
    print('[INFO] --> Computing Optical Flow Video for the video: "%s"' % video.name)
    video_path = video.path
    flow = []
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    vidcap = cv2.VideoCapture(video_path)
    success,frame1 = vidcap.read()
    bins = np.linspace(-20, 20, num=256)
    prev = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
    vid_len = get_video_length(video_path)
    for _ in range(0,vid_len-1):
        success, frame2 = vidcap.read()
        curr = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) 
        curr_flow = TVL1.calc(prev, curr, None)
        assert(curr_flow.dtype == np.float32)
        
        #Truncate large motions
        curr_flow[curr_flow >= 20] = 20
        curr_flow[curr_flow <= -20] = -20
        
        #digitize and scale to [-1;1]
        curr_flow = np.digitize(curr_flow, bins)
        curr_flow = (curr_flow/255.)*2 - 1
        
        #cropping the center
        curr_flow = curr_flow[8:232, 48:272]  
        flow.append(curr_flow)
        prev = curr
    vidcap.release()
    flow = np.asarray([np.array(flow)])
    print('[INFO] --> Shape of the computed flow: ', flow.shape)
    return flow


def get_video_length(video_path):
    _, ext = os.path.splitext(video_path)
    if not ext in VIDEO_FILE_PREFIX:
        raise ValueError('[ERROR] --> Extension "%s" not supported' % ext)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        raise ValueError("[ERROR] --> Could not open the file.\n{}".format(video_path))
    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
    length = int(cap.get(CAP_PROP_FRAME_COUNT))
    cap.release()
    return length