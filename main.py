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
        blurred_img = cv2.GaussianBlur(img,(7,7),0)


        # THRESH_OTSU determines the best threshold value based on the image histogram then uses this threshold
        #ret, threshold_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # local threshold
        threshold_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

        # print the used threshold value (if pixel < ret: pixel is set to 0 else: 255)
        inverted_img = cv2.bitwise_not(threshold_img)
        # Apply Canny edge detection
        canny = cv2.Canny(img, 50, 240)
        #Adding Canny and Threshold value
        adding = cv2.add(canny,inverted_img)
        #print the width and the lenght
       # print inverted_img.shape
        #crop the picture 
       # cropped = inverted_img[720:1450, 100:1240]

        # save the images
        result_filename = join(train_result_path, f)
        cv2.imwrite(result_filename, adding)

        # display images
        # cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
        # cv2.imshow('image', inverted_img)

    # remove the break if you are not displaying the image
    # break
cv2.waitKey(0)
cv2.destroyAllWindows()

