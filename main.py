from os import listdir, makedirs
from os.path import isfile, join, exists
import imutils
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import matplotlib.pyplot as plt

train_images_path = '../train'
test_images_path = '../test'

train_result_path = '../train_result'
cropped_path='../cropped'

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

        # THRESH_OTSU determines the best threshold value based on the image histogram then uses this threshold
        #ret, threshold_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edged = cv2.Canny(blurred, 130, 300)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # ensure that at least one contour was found
        if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            rect = None
            # loop over the sorted contours
            for c in cnts:
            # approximate the contour
                (x, y, w, h) = cv2.boundingRect(c)
                #print (x, y, w, h)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # if our approximated contour has four points,
                # then we can assume we have found the paper
                if len(approx) == 4 and w >= 900 and h >= 500:
                    max_width = w
                    max_height = h
                    rect = approx
                    break

            if rect is not None:
                paper = four_point_transform(edged, rect.reshape(4, 2))
                image = four_point_transform(img, rect.reshape(4, 2))
                backtorgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
                closing = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel)

                kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                erosion = cv2.erode(closing, kernel2, iterations=1)

                questions_cols = []
                questions_cols.append(erosion[70:665, 90:270])
                questions_cols.append(erosion[70:665, 418:598])
                questions_cols.append(erosion[70:665, 745:925])

                for questions_col in questions_cols:
                    circles = cv2.HoughCircles(questions_col, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=2, maxRadius=20)
                    if circles is not None:
                        for circle in circles[0]:
                            print circle
                            #cv2.circle(backtorgb, (circle[0], circle[1]), circle[2], (0, 0, 255), -1)
                        fig, ax = plt.subplots()
                        im = ax.imshow(backtorgb, interpolation='none')
                        #plt.show()
                        #cv2.imshow('image',backtorgb)
                        # save the images
                        result_filename = join(train_result_path, f)
                        cv2.imwrite(result_filename, backtorgb)

                        '''
                        f1 = join(train_result_path, '1.png')
                        cv2.imwrite(f1, questions_cols[0])
                        f2 = join(train_result_path, '2.png')
                        cv2.imwrite(f2, questions_cols[1])
                        f3 = join(train_result_path, '3.png')
                        cv2.imwrite(f3, questions_cols[2])
                        '''

cv2.waitKey(0)
