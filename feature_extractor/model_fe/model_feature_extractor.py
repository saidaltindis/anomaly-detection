'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import numpy as np
import argparse

from i3d_inception import Inception_Inflated3d

NUM_FRAMES = 79
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 2

SAMPLE_DATA_PATH = {
    'rgb' : 'data/rgb2.npy',
    'flow' : 'data/flow2.npy'
}

LABEL_MAP_PATH = 'data/label_map.txt'

def main(args):

    # build model for RGB data
    # and load pretrained weights (trained on imagenet and kinetics dataset)
    rgb_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
    
    rgb_model.summary()

        # load RGB sample (just one example)
    rgb_sample = np.load(SAMPLE_DATA_PATH['rgb'])
        
    rgb_logits = rgb_model.predict(rgb_sample)


            # build model for optical flow data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
    flow_model = Inception_Inflated3d(
                include_top=False,
                weights='flow_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)


    flow_model.summary()

    # load flow sample (just one example)
    flow_sample = np.load(SAMPLE_DATA_PATH['flow'])
        
    flow_logits = flow_model.predict(flow_sample)


    # produce final model logits
    print('RGB LOGITS:' + str(type(rgb_logits)))
    print('RGB LOGITS:' + str(rgb_logits.shape))
    print('FLOW LOGITS:' + str(type(flow_logits)))
    print('FLOW LOGITS:' + str(flow_logits.shape))
    return


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-type', 
        help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).', 
        type=str, choices=['rgb', 'flow', 'joint'], default='joint')

    parser.add_argument('--no-imagenet-pretrained',
        help='If set, load model weights trained only on kinetics dataset. Otherwise, load model weights trained on imagenet and kinetics dataset.',
        action='store_true')


    args = parser.parse_args()
    main(args)
