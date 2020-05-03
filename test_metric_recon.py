import numpy as np
import metric_recon as mr
import cv2
import unittest


class TestMyCVlib(unittest.TestCase):


    def test_compute_epipole(self):
        img_path1 = '/home/garrett/computervision/hw2/data/graf1.png'
        img_path2 = '/home/garrett/computervision/hw2/data/graf2.png'
        img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
        gray1 = cv2.imread(img_path1, 0)
        gray2 = cv2.imread(img_path2, 0)
        
        F, matches1, matches2 = mr.compute_F(gray1,gray2)

        epi1 = mr.compute_epipole(F)
        epi2 = mr.compute_epipole(F.transpose())
        self.assertTrue(np.linalg.norm(np.dot(F, epi1)) < 1e-10, 'Should equal zero')
        self.assertTrue(np.linalg.norm(np.dot(epi2, F)) < 1e-10, 'Should equal zero')

    def test_estimate_projection(self):
        img_path1 = '/home/garrett/computervision/hw2/data/graf1.png'
        img_path2 = '/home/garrett/computervision/hw2/data/graf2.png'
        img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
        gray1 = cv2.imread(img_path1, 0)
        gray2 = cv2.imread(img_path2, 0)
        
        F, matches1, matches2 = mr.compute_F(gray1,gray2)

        epi1 = mr.compute_epipole(F)
        epi2 = mr.compute_epipole(F.transpose())

        P2 = mr.estimate_projection(epi2,F)

        self.assertTrue(P2.shape[0] == 3, 'Should have 3 rows')
        self.assertTrue(P2.shape[1] == 4, 'Should have 4 cols')

        for i in range(epi2.shape[0]):
            self.assertEqual(P2[i,3], epi2[i], 'Last col should be epipole')



    '''
    def test_np_cross(self):
        x = np.array([1, 2, 3])
        A = np.arange(9).reshape(3,3)
        y1 = np.cross(x,A)

        x_hat = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]],[-x[1], x[0], 0]])
        y2 = x_hat @ A
        print(y1)
        print(y2)
        for i in range(y1.shape[0]):
            for j in range(y1.shape[1]):
                self.assertEquals(y1[i,j], y2[i,j], 'Should be equal')
    '''
       

if __name__ == '__main__':
    unittest.main()
