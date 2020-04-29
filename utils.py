import numpy as np
from PIL import Image

def DepthNorm(x, maxDepth):
    return maxDepth / x

def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth

def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append( resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True ) )

    return np.stack(scaled)

def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open( file ), dtype=float) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)

def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:,:,0]
    return np.stack((i,i,i), axis=2)

def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize

    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)

    all_images = []

    for i in range(outputs.shape[0]):
        imgs = []

        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if is_colormap:
            rescaled = outputs[i][:,:,0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            imgs.append(plasma(rescaled)[:,:,:3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)

    return skimage.util.montage(all_images, multichannel=True, fill=(0,0,0))

def save_images(filename, outputs, inputs=None, gt=None, is_colormap=True, is_rescale=False):
    montage =  display_images(outputs, inputs, is_colormap, is_rescale)
    im = Image.fromarray(np.uint8(montage*255))
    im.save(filename)

def load_test_data(test_data_zip_file='nyu_test.zip'):
    print('Loading test data...', end='')
    import numpy as np
    from data import extract_zip
    data = extract_zip(test_data_zip_file)
    from io import BytesIO
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    print('Test data loaded.\n')
    return {'rgb':rgb, 'depth':depth, 'crop':crop}

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10

def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetDepths = []

    for i in range(N//bs):
        x = rgb[(i)*bs:(i+1)*bs,:,:,:]

        # Compute results
        true_y = depth[(i)*bs:(i+1)*bs,:,:]
        pred_y = scale_up(2, predict(model, x/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2, predict(model, x[...,::-1,:]/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y = pred_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y_flip = pred_y_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]

        # Compute errors per image in batch
        for j in range(len(true_y)):
            predictions.append(   (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j]))   )
            testSetDepths.append(   true_y[j]   )

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(predictions, testSetDepths)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

    return e

def convert_to_base_frame(pointcloud, transforms, start_index, desired_index):
    # Transforming pointcloud to pose of desired frame
    # pointcloud - Nx3 matrix of points
    # transforms - list of 3x3 essential matrices relating the 3d points in one frame to the next
        # it is important that this is sequential and indexed by frame ie: transforms[2] is the transform from frame 2->3
    # start_index - the frame # of pose to be transformed
    # desired_index - the frame to be transformed into
    # Returns - a pointcloud in the new frame and the resultant transform

    if desired_index == start_index:
        transform = np.eye(4)
    elif desired_index > start_index:
        transform = transforms[start_index]
        for idx in range(start_index + 1, desired_index):
            transform = np.matmul(transforms[idx], transform)
    else:
        transform = np.linalg.inv(transform)
        for idx in range(start_index - 1, desired_index, -1):
            transform = np.matmul(np.linalg.inv(transforms[idx]), transform)

    # homogenous form
    pointcloud = np.hstack((pointcloud,np.ones(pointcloud.shape[0]).reshape(-1,1)))

    # Nx4:  x2^T = x1^T * A^T
    new_cloud = pointcloud @ transform.transpose()
     
    return new_cloud[:,:-1], transform.transpose()


def transforms(images, masks):
    '''Takes in a list of images and an array of their respective masks.

    Args:
        - images (List of 2D unit8 arrays): List of sequential grayscale images
        - masks (3D Nxrxc binary array): Array of masks for N frames

    Returns:
        - tfs (3D Nx4x4 float array): each 4x4 is the homogenous transformation
        between the neighboring images. Ex. tfs[i,:,:] is the transformation
        [R t; 0 0 0 1] such that it's the change of basis from frame 1 to frame2

    '''
    # preallocate
    tfs = np.zeros([len(images)-1,4,4])

    for i in range(len(images)-1):
        frame1 = images[i]
        frame2 = images[i+1]
        mask1 = masks[i]
        mask2 = masks[i+1]

        # Find transformation between frames
        tfs[i,:,:]= compute_tf(frame1, frame2, mask1, mask2)

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
