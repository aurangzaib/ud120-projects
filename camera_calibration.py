import cv2 as cv
import numpy as np


def get_undisorted_image(nx, ny, image, corner, found, obj_pts, obj_pt, img_pts):
    if found is True:
        img_pts.append(corner)
        obj_pts.append(obj_pt)
        draw_pts = np.copy(image)
        cv.drawChessboardCorners(image=draw_pts, patternSize=(nx, ny), corners=corner, patternWasFound=found)
        ret, camera_matrix, dist_coef, rot_vector, trans_vector = cv.calibrateCamera(objectPoints=obj_pts,
                                                                                     imagePoints=img_pts,
                                                                                     imageSize=image.shape[0:2],
                                                                                     cameraMatrix=None,
                                                                                     distCoeffs=None)
        undistorted_image = cv.undistort(src=image,
                                         cameraMatrix=camera_matrix,
                                         distCoeffs=dist_coef,  # undistorted coef
                                         dst=None,
                                         newCameraMatrix=camera_matrix)

        cv.imshow("found corners", draw_pts)
        cv.imshow("undistorted", undistorted_image)


# print(gray.shape[:-1])    --> give last
# print(gray.shape[::-1])   --> give all except last

image = cv.imread("/Users/siddiqui/Downloads/chess-iphone.jpg")
cv.imshow("original", image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
nx, ny, channels = 9, 6, 3

# img_pts --> 2D
# obj_pts --> 3D in real world
img_pts, obj_pts = [], []

# to create a matrix of 4x5 --> np.mgrid[0:4, 0:5]
obj_pt = np.zeros(shape=(nx * ny, channels), dtype=np.float32)
obj_pt[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

found, corner = cv.findChessboardCorners(image=gray, patternSize=(nx, ny))
print("# of corners: {}".format(len(corner)))

get_undisorted_image(nx, ny, image, corner, found, obj_pts, obj_pt, img_pts)
cv.waitKey()
