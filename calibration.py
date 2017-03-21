"""calibration.py: Calibration the cameras and save the calibration results."""

__author__ = "Junsheng Fu"
__email__ = "junsheng.fu@yahoo.com"
__date__ = "March 2017"

import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
from os import path


def calibrate_camera(nx, ny, basepath):
    """

    :param nx: number of grids in x axis
    :param ny: number of grids in y axis
    :param basepath: path contains the calibration images
    :return: write calibration file into basepath as calibration_pickle.p
    """

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(path.join(basepath, 'calibration*.jpg'))

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imshow('input image',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()


    # calibrate the camera
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we don't use rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    destnation = path.join(basepath,'calibration_pickle.p')
    pickle.dump( dist_pickle, open( destnation, "wb" ) )
    print("calibration data is written into: {}".format(destnation))

    return mtx, dist


def load_calibration(calib_file):
    """

    :param calib_file:
    :return: mtx and dist
    """
    with open(calib_file, 'rb') as file:
        # print('load calibration data')
        data= pickle.load(file)
        mtx = data['mtx']       # calibration matrix
        dist = data['dist']     # distortion coefficients

    return mtx, dist


def undistort_image(imagepath, calib_file, visulization_flag):
    """ undistort the image and visualization

    :param imagepath: image path
    :param calib_file: includes calibration matrix and distortion coefficients
    :param visulization_flag: flag to plot the image
    :return: none
    """
    mtx, dist = load_calibration(calib_file)

    img = cv2.imread(imagepath)

    # undistort the image
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_undistRGB = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)

    if visulization_flag:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(imgRGB)
        ax1.set_title('Original Image', fontsize=30)
        ax1.axis('off')
        ax2.imshow(img_undistRGB)
        ax2.set_title('Undistorted Image', fontsize=30)
        ax2.axis('off')
        plt.show()

    return img_undistRGB


if __name__ == "__main__":

    nx, ny = 9, 6  # number of grids along x and y axis in the chessboard pattern
    basepath = 'camera_cal/'  # path contain the calibration images

    # calibrate the camera and save the calibration data
    calibrate_camera(nx, ny, basepath)
