#!/usr/bin/env python3
'''
    Author: Garrett Ung, gung1@jhu.edu
'''

import numpy as np
import cv2
import time

def essential(gray_img1, gray_img2, mask1=None, mask2=None):
    '''Takes in two gray images, their masks, and outputs their essential matrix.

    Args:
        - gray_img1 (2D unit8 array): grayscale image 1
        - gray_img2 (2D unit8 array): grayscale image 2
        - mask1 (2D binary array?): object sementation mask for image 1
        - mask2 (2D binary array?): object sementation mask for image 2
    Returns:
        - E (3x3 float array): essential matrix

    '''
    sift = cv2.xfeatures2d.SIFT_create()

    # SIFT keypoints and descriptors for each masked image
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)
    print('kp1 type: ', type(kp1))
    print('kp2 type: ', type(kp2))
    print('des1 type: ', type(des1))
    print('des2 type: ', type(des2))
    print('kp1 : ', (kp1[0]))
    print('kp2 : ', (kp2[0]))
    print('des1 : ', (des1[0]))
    print('des2 : ', (des2[0]))
    print(des1[0].dtype)

    # Find corresponding matches between keypoints
    # matches1 and matches2 are the actual points, not the indices
    matches1, matches2 = find_matches(kp1, kp2, des1, des2)

    return kp1, kp2


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

    # Use FLANN based matcher
    # parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)         # # of times tree should be recursively traversed. Higher takes longer

    matcher = cv2.FlannBasedMatcher(index_params,search_params)

    matches = matcher.knnMatch(des1,des2,k=2)

    good_matches = []
    good_pts1 = []          # list of tuples
    good_pts2 = []          # list of corresponding tuples

    print('match : ', matches[0])

    # Determine good matches using ratio test
    for m, n in matches:
        if m.distance < ratio_test_thresh*n.distance:
            good_matches.append(m)
            good_pts2.append(kp2[m.trainIdx].pt)
            good_pts1.append(kp1[m.queryIdx].pt)
    print('good_matches: ', good_matches[0])
    print(good_matches[0].trainIdx)
    print(good_matches[0].queryIdx)
    print('good_points1: ', good_pts1[0])
    print('good_points1: ', good_pts2[0])


    return good_pts1, good_pts2


def test_main():
    test_essential()


def test_essential():
    img_path1 = 'C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/computervision/hw2/data/graf1.png'
    img_path2 = 'C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/computervision/hw2/data/graf2.png'
    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    gray1 = cv2.imread(img_path1, 0)
    gray2 = cv2.imread(img_path2, 0)

    kp1, kp2 = essential(gray1, gray2)

    outimg1=cv2.drawKeypoints(gray1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/test/img1_keypoints.png',outimg1)
    outimg2=cv2.drawKeypoints(gray2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/test/img2_keypoints.png',outimg2)



if __name__ == '__main__':
    tic = time.process_time()       # debugging
    test_main()
    toc = time.process_time()       # debugging
    print('Total computation time = '+str(1000*(toc-tic))+'ms')     # debugging
