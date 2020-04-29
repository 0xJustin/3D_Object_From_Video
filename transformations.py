#!/usr/bin/env python3
'''
    Author: Garrett Ung, gung1@jhu.edu
'''

import numpy as np
import cv2
import time
import scipy.linalg

def transforms(images, masks=None):
    '''Takes in a list of images and their respective masks.
        Can change to arrays too

    Args:
        - images (List of 2D unit8 arrays): List of sequential grayscale images
        - masks (List of 2D binary arrays): List of sequential masks

    Returns:
        - tfs (3D 4x4xN float array): each 4x4 is the homogenous transformation
        between the neighboring images. Ex. tfs[:,:,1] is the transformation
        [R t; 0 0 0 1] such that it's the change of basis from frame 1 to frame2

    '''
    # preallocate
    tfs = np.zeros([4,4,len(images)-1])

    for i in range(len(images)-1):
        frame1 = images[i]
        frame2 = images[i+1]
        mask1 = masks[i]
        mask2 = masks[i+1]

        # Find transformation between frames
        tfs[:,:,i]= compute_tf(frame1, frame2, mask1, mask2)

    return tfs


def compute_tf(gray_img1, gray_img2, mask1=None, mask2=None):
    '''Takes in two gray images, their masks, and outputs their essential matrix.

    Args:
        - gray_img1 (2D unit8 array): grayscale image 1
        - gray_img2 (2D unit8 array): grayscale image 2
        - mask1 (2D binary array?): object sementation mask for image 1
        - mask2 (2D binary array?): object sementation mask for image 2
    Returns:
        - tf (4x4 float array): transformation [R t; 0 0 0 1] such that
            it's the change of basis from frame 1 to frame2

    '''
    # ASSUMPTIONS:
    # For now assuming that center pixel is the principle point
    # Also assuming focal length
    # Assuming images have the same size
    ppmm = 11       # assumed pixel/mm, near iphone pixel density
    f = 40            # assumed focal length (mm)
    ppx = int(gray_img1.shape[1]) / 2
    ppy = int(gray_img1.shape[0]) / 2
    # intrinsic parameter matrix
    K = np.array([[f*ppmm, 0, ppx],[0, f*ppmm, ppy],[0,0,1]])

    # SIFT keypoints and descriptors for each masked image
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    # Find corresponding matches between keypoints
    # matches1 and matches2 are the actual points, not the indices
    matches1, matches2 = find_matches(kp1, kp2, des1, des2)

    # Convert to numpy arrays
    matches1 = np.array(matches1).astype(float)
    matches2 = np.array(matches2).astype(float)

    # Compute the essential matrix, use RANSAC to handle outliers
    # inliers is an array 1=inlier, 0=outlier
    # NOTE that E is transpose of one from class such that ((x2^T)E*x1)
    E, inliers = cv2.findEssentialMat(matches1, matches2, K, method=cv2.RANSAC,
            prob=0.995,threshold=1.0)

    # Only the inlier matches
    matches1 = matches1[inliers.ravel() == 1]
    matches2 = matches2[inliers.ravel() == 1]

    # Recover the homogenous transformation
    points, R, t, mask = cv2.recoverPose(E,matches1, matches2, K)
    tf = np.hstack((R,t.reshape(-1,1)))
    tf = np.vstack((tf,np.array([0, 0, 0, 1]).reshape(1,-1)))

    return tf


def find_matches(kp1, kp2, des1, des2):
    '''Finds matches between feature points in two images. Does NOT remove outliers.

    Args:
        - kp1 (List of Keypoint objects) - for image 1
        - kp2 (List of Keypoint objects) - for image 2
        - des1 (2D float array) - each row is a point's histogram descriptor for img1
        - des2 (2D float array) - each row is a point's histogram descriptor for img2
    Returns:
        - good_pts1 (List of tuples) - (x, y) good matches, indices align with good_pts2
        - good_pts2 (List of tuples) - (x, y) good matches, indices align with good_pts1
    '''
    # Parameters
    ratio_test_thresh = 0.8

    # Use FLANN based matcher to find matching keypoints --------------------
    # Parameters:
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)         # # of times tree should be recursively traversed. Higher takes longer

    matcher = cv2.FlannBasedMatcher(index_params,search_params)
    matches = matcher.knnMatch(des1,des2,k=2)

    # Determine good matches using ratio test ----------------------------
    good_pts1 = []          # list of tuples
    good_pts2 = []          # list of corresponding tuples
    for m, n in matches:
        if m.distance < ratio_test_thresh*n.distance:
            good_pts2.append(kp2[m.trainIdx].pt)
            good_pts1.append(kp1[m.queryIdx].pt)

    return good_pts1, good_pts2


# Testing methods ------------------------------------------------------------
def test_main():
    test_compute_tf()


def test_compute_tf():
    img_path1 = 'C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/computervision/hw2/data/graf1.png'
    img_path2 = 'C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/computervision/hw2/data/graf2.png'
    img_path3 = 'C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/computervision/hw2/data/graf3.png'
    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    img3 = cv2.imread(img_path3, cv2.IMREAD_COLOR)
    gray1 = cv2.imread(img_path1, 0)
    gray2 = cv2.imread(img_path2, 0)
    gray3 = cv2.imread(img_path2, 0)

    images = [gray1, gray2, gray3]
    masks = [None, None, None]
    tf = transforms(images, masks)

    for i in range(tf.shape[2]):
        print(tf[:,:,i])

    #outimg1=cv2.drawKeypoints(gray1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite('C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/test/img1_keypoints.png',outimg1)
    #outimg2=cv2.drawKeypoints(gray2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite('C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/test/img2_keypoints.png',outimg2)



if __name__ == '__main__':
    tic = time.process_time()       # debugging
    test_main()
    toc = time.process_time()       # debugging
    print('Total computation time = '+str(1000*(toc-tic))+'ms')     # debugging
