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
import time
import imutils

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

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

    print('Running model...')
    outputs = predict(model, inputs)
    print('Done.')

    return outputs

def segmentation(images):  
    # load the class label names
    CLASSES = open("enet-cityscapes/enet-classes.txt").read().strip().split("\n")
    
    # initialize a list of colors to represent each class label in mask
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
    	dtype="uint8")
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
    
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNet("enet-cityscapes/enet-model.net")

    image_width = 500
    masks = []

    print("[INFO] segmenting images")
    for image in images:
        image = imutils.resize(image, width=image_width)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0,
        	swapRB=True, crop=False)
        
        # perform a forward pass using the segmentation model
        net.setInput(blob)
        start = time.time()
        output = net.forward()
        end = time.time()
        #print("[INFO] inference took {:.4f} seconds".format(end - start))
    
        # Get number of classes and dimensions of mask from shape of output array
        (numClasses, height, width) = output.shape[1:4]
        # Output class ID map
        classMap = np.argmax(output[0], axis=0)
        # given the class ID map, map each of the class IDs to corresponding color
        mask = COLORS[classMap]
        
        # resize the mask and class map such that its dimensions match the
        # original size of the input image
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
        	interpolation=cv2.INTER_NEAREST)
        
        masks.append(mask)
        
    return masks
    
  

def main():
    vid_path = 'data/car_donut.mp4'
    images = images_from_video(vid_path)
    
    num_depth_img_samples = 50
    image_samples = [images[int(len(images)*i/num_depth_img_samples)] for i in range(num_depth_img_samples)]
    #resized_image_samples = [cv2.resize(image, (640, 480)) for image in image_samples]
    #depth_images = depth_estimation(resized_image_samples)
    
    masks = segmentation(image_samples)

    for i in range(num_depth_img_samples):
        #cv2.imshow('image', cv2.resize(image_samples[i], (320, 240)))
        #cv2.waitKey(0)
        #cv2.imshow('image', depth_images[i])
        #cv2.waitKey(0)
        cv2.imshow('Segmented mask', masks[i])
        cv2.waitKey(0)

if __name__ == '__main__':
  main()