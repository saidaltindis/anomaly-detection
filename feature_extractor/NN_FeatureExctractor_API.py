import numpy as np
import argparse
from I3D import Inception_Inflated3d


def extractNNFeatures(rgb, opticalFlow):
    print("[INFO] --> Retrieving features from RGB Stream")
    rgb_stream = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(rgb.shape[1], rgb.shape[2], rgb.shape[3], rgb.shape[4]),
                )
        
    NN_RGB = rgb_stream.predict(rgb)

    print("[INFO] --> Retrieving features from Optical Flow Stream")
    flow_stream = Inception_Inflated3d(
                include_top=False,
                weights='flow_imagenet_and_kinetics',
                input_shape=(opticalFlow.shape[1], opticalFlow.shape[2], opticalFlow.shape[3], opticalFlow.shape[4]),
                )

    NN_OF = flow_stream.predict(opticalFlow)


    return NN_RGB, NN_OF