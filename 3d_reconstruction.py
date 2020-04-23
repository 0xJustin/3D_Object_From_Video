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

# press q to quit playing video if it is slow




def main():
    
    #video path
    vid_path = 'data/video1.mp4'
    cap = cv2.VideoCapture(vid_path)
    
    #num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while(cap.isOpened()):
    #for i in range(num_frames):
        
        # get frame of video - default returns sequential frames
        # uncomment to get frame by index using for loop
        #cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # uncomment below to get frame by msec
        #cap.set(cv2.CAP_PROP_POS_MSEC, 200) 
        ret, frame = cap.read()
        
        # check that a frame was returned, if not break
        if not ret:
            break;
        else:
            # frame holds the video frame for processing
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            # additional functions here:
        
        
        
        
        
        
        
            cv2.imshow('frame',gray_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # close video stream and windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
  main()