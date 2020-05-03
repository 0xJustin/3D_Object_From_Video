import numpy as np
import time
import transformations
import cv2


# Testing methods ------------------------------------------------------------
def test_main():
    mask = np.load('/home/garrett/Downloads/mask_thresh_marine.npy')
    print(mask.shape)
    print(mask.T.shape)
    test_compute_tf()


def test_compute_tf():
    img_path1 = '/home/garrett/computervision/hw2/data/graf1.png'
    img_path2 = '/home/garrett/computervision/hw2/data/graf2.png'
    img_path3 = '/home/garrett/computervision/hw2/data/graf3.png'
    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    img3 = cv2.imread(img_path3, cv2.IMREAD_COLOR)
    gray1 = cv2.imread(img_path1, 0)
    gray2 = cv2.imread(img_path2, 0)
    gray3 = cv2.imread(img_path2, 0)

    images = [gray1, gray2, gray3]
    masks = np.array([[None], [None], [None]])
    tf = transformations.transforms(images, masks)

    print(tf[0,:,:])
    print(tf[1,:,:])
    print(tf[0])



    #outimg1=cv2.drawKeypoints(gray1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite('C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/test/img1_keypoints.png',outimg1)
    #outimg2=cv2.drawKeypoints(gray2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imwrite('C:/Users/garre/OneDrive - Johns Hopkins University/2020_Spring (M)/test/img2_keypoints.png',outimg2)



if __name__ == '__main__':
    tic = time.process_time()       # debugging
    test_main()
    toc = time.process_time()       # debugging
    print('Total computation time = '+str(1000*(toc-tic))+'ms')     # debugging