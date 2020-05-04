#!/usr/bin/env python3

import numpy as np
import cv2
import time
import scipy.linalg

def compute_epipole(F):
    '''Computes the position of the epipole in the specified image, given the
        fundamental matrix.

        In the right image, the epipole satisfies F*e = 0
        In the left image, the epipole satisfies F^T*e = 0

    Args: 
        - F (3x3 float array): fundamental matrix (x2^T * F * x1 = 0)

    Returns:
        - epi ()
    '''
    # Solve F*e = 0 least squared solution
    # Use SVD
    U,S,Vh = scipy.linalg.svd(F)
    V = Vh.transpose()
    epi = V[:,-1]
    epi = epi/epi[2]        # divide to get [u, v, 1]

    return epi


def compute_plane_infinity():
    ''' Computes the plane at infinity. Not sure if this is necessary


    '''



def estimate_projection(epi2, F):
    ''' Compute the projection matrix from the fundamental matrix.

    Args:
        epi (3x1 float array) : the epipole in homogeneous coords 
            corresponding to first camera in second image
        F (3x3 float array): fundamental matrix relating features in the images
    Returns:
        P2 (3x4 float array): Projection matrix for image 2

    '''
    P2 = np.zeros([3,4])

    # From https://www.cs.unc.edu/~marc/tutorial/node62.html
    epi2_hat = np.array([[0, -epi2[2], epi2[1]], [epi2[2], 0, -epi2[0]],[-epi2[1], epi2[0], 0]])
    P2[:,:3] = epi2_hat @ F
    P2[:,3] = epi2

    return P2



def compute_F(gray_img1, gray_img2, mask1=None, mask2=None):
    '''Takes in two gray images, their masks, and outputs their fundamental matrix.

    Args:
        - gray_img1 (2D unit8 array): grayscale image 1
        - gray_img2 (2D unit8 array): grayscale image 2
        - mask1 (2D binary array?): object sementation mask for image 1
        - mask2 (2D binary array?): object sementation mask for image 2
    Returns:
        - F (3x3 float array): fundamental matrix (x2^T * F * x1 = 0)

    '''

    # SIFT keypoints and descriptors for each masked image
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_img1, mask1)
    kp2, des2 = sift.detectAndCompute(gray_img2, mask2)

    # Find corresponding matches between keypoints
    # matches1 and matches2 are the actual points, not the indices
    matches1, matches2 = find_matches(kp1, kp2, des1, des2)

    # Convert to numpy arrays
    matches1 = np.array(matches1).astype(float)
    matches2 = np.array(matches2).astype(float)

    # Compute the fundamental matrix, use RANSAC to handle outliers
    # inliers is an array 1=inlier, 0=outlier
    # NOTE that E is transpose of one from class such that ((x2^T)F*x1)
    F, inliers = cv2.findFundamentalMat(matches1, matches2, method=cv2.RANSAC,
            ransacReprojThreshold=1.0, confidence=0.995)

    # Only the inlier matches
    matches1 = matches1[inliers.ravel() == 1]
    matches2 = matches2[inliers.ravel() == 1]


    return F, matches1, matches2


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

def get_feature_pairs(img1, img2, mask1=None, mask2=None):
    '''Takes in two images, their masks, and outputs their fundamental matrix.

    Args:
        - gray_img1 (2Dx3 unit8 array): image 1
        - gray_img2 (2Dx3 unit8 array): image 2
        - mask1 (2D binary array?): object sementation mask for image 1
        - mask2 (2D binary array?): object sementation mask for image 2
    Returns:

    '''
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT keypoints and descriptors for each masked image
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, mask1)
    kp2, des2 = sift.detectAndCompute(gray2, mask2)

    # Find corresponding matches between keypoints
    # matches1 and matches2 are the actual points, not the indices
    matches1, matches2 = find_matches(kp1, kp2, des1, des2)

    # Convert to numpy arrays
    matches1 = np.array(matches1).astype(float)
    matches2 = np.array(matches2).astype(float)

    # Compute the fundamental matrix, use RANSAC to handle outliers
    # inliers is an array 1=inlier, 0=outlier
    # NOTE that E is transpose of one from class such that ((x2^T)F*x1)
    F, inliers = cv2.findFundamentalMat(matches1, matches2, method=cv2.RANSAC,
            ransacReprojThreshold=2.0, confidence=0.995)

    # Only the inlier matches
    matches1 = matches1[inliers.ravel() == 1]
    matches2 = matches2[inliers.ravel() == 1]

    return matches1, matches2

def get_features(images, masks=None):
    '''Output list of points, feature#, image# and image filename from a sequence of images


    '''
    # First 2 images --------------------------------------------------
    matches1, matches2 = get_feature_pairs(images[0], images[1])


    features = np.zeros([images.shape[0],4, matches1.shape[0]])

    # points
    features[0,0:2,:] = matches1.transpose()
    features[1,0:2,:] = matches2.transpose()

    
    # feature ID#
    features[0,2,:] = np.arange(1,matches1.shape[0]+1)
    features[1,2,:] = np.arange(1,matches2.shape[0]+1)

    # Image ID #
    features[0,3,:] = np.ones(matches1.shape[0])
    features[1,3,:] = 2*np.ones(matches2.shape[0])

    for i in range(1, images.shape[0]-1):
        matches1, matches2 = get_feature_pairs(images[i], images[i+1])
        #print(matches1)
        #print(matches2)

        # for each match
        for j in range(matches1.shape[0]):
            # Find if the matches were already paired before
            commonx = (features[i-1,0,:] - matches1[j,0] < 1e-1)
            commony = (features[i-1,1,:] - matches1[j,1] < 1e-1)

            # boolean -- both x and y coordinates have to match
            common = commonx * commony

            #print(common)
            common_ind = np.nonzero(common)
            #print(common_ind)


            if common_ind[0].size == 0:
                continue

            # Add common feature in next image to array
            features[i+1,0:2,common_ind] = matches2[j,:]
            # Add image ID
            features[i+1,3,common_ind] = i+1
            # Add feature ID
            features[i+1,2,common_ind] = common_ind

    return features




    
