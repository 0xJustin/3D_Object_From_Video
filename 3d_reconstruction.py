'''
CV Final Project - 3d Reconstruction from video
 Charlie Watkins
 Garret Ung
 Justin Joyce
 Luke Robinson
 Will David
 
'''

import numpy as np
import cv2
import os
import glob
import argparse
import matplotlib
from keras.models import load_model
from matplotlib import pyplot as plt

from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

def images_from_video(vid_path):
    images = []
    
    # press q to quit playing video if it is slow
    cap = cv2.VideoCapture(vid_path)
    
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #while(cap.isOpened()):
    for i in range(num_frames):
        
        # get frame of video - default returns sequential frames
        # uncomment to get frame by index using for loop
        #cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # uncomment below to get frame by msec
        # cap.set(cv2.CAP_PROP_POS_MSEC, 2000) 
        ret, frame = cap.read()
        
        # check that a frame was returned, if not break
        if not ret:
            break;
        else:
            # frame holds the video frame for processing
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            # additional functions here:
            images.append(frame)

            # cv2.imshow('frame',gray_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    # close video stream and windows
    cap.release()
    cv2.destroyAllWindows()

    return images

def depth_estimation(images):
    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    # Load model
    print('Loading model...')
    model = load_model("nyu.h5", custom_objects=custom_objects, compile=False)
    print('Model loaded ({0}).'.format("nyu.h5"))

    # Input images
    #inputs = load_images( glob.glob('data/random_car_2.jpg') )
    inputs = np.array(images)

    outputs = predict(model, inputs)

    return outputs

def main():
    vid_path = 'data/video1.mp4'
    images = images_from_video(vid_path)
    
    image_size = (640, 480)
    num_img_samples = 10
    image_samples = [images[int(len(images)*i/num_img_samples)] for i in range(num_img_samples)]
    resized_images = [cv2.resize(image, image_size) for image in image_samples]
    depth_images = depth_estimation(resized_images)

    for i in range(num_img_samples):
        cv2.imshow('image', cv2.resize(image_samples[i], (320, 240)))
        cv2.waitKey(0)
        cv2.imshow('image', depth_images[i])
        cv2.waitKey(0)

if __name__ == '__main__':
  main()