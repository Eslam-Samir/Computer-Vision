from os import listdir, makedirs
from os.path import isfile, join, exists
import cv2
import numpy as np

train_images_path = '../train'
test_images_path = '../test'

train_result_path = '../train_result'

if not exists(train_result_path):
    makedirs(train_result_path)

# loop through all of the files in the directory
for f in listdir(train_images_path):
    # the total path of the current file name
    filename = join(train_images_path, f)
    # check if it's a file
    if isfile(filename):
        print filename
        # read the image file
        img = cv2.imread(filename, 0)
        # apply gaussian blur to remove random dots
        blurred_img = cv2.GaussianBlur(img,(3,3),0)
        # THRESH_OTSU determines the best threshold value based on the image histogram then uses this threshold
        ret, threshold_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # print the used threshold value (if pixel < ret: pixel is set to 0 else: 255)
        inverted_img = cv2.bitwise_not(threshold_img)
        print ret

        # save the images
        result_filename = join(train_result_path, f)
        cv2.imwrite(result_filename, inverted_img)

        # display images
        # cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
        # cv2.imshow('image', inverted_img)

    # remove the break if you are not displaying the image
    # break
cv2.waitKey(0)
cv2.destroyAllWindows()

