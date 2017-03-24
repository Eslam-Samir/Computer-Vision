from os import listdir, makedirs
from os.path import isfile, join, exists
import imutils
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import argparse

train_images_path = '../train'
test_images_path = '../test'

train_result_path = '../train_result'
cropped_path='../cropped'
'''
if not exists(train_result_path):
    makedirs(train_result_path)
	
if not exists(cropped_path):
    makedirs(cropped_path)
for f in listdir(train_images_path):
	filename = join(train_images_path, f)
	img = cv2.imread(filename,0)
	cropped = img[700:,:]
	result_filename = join(cropped_path, f)
	cv2.imwrite(result_filename, cropped)

'''
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
        minLineLength = 30
        maxLineGap = 10
        lines = cv2.HoughLinesP(edged,1,np.pi/180,15,minLineLength,maxLineGap)
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        docCnt = None
        rectangles=[]
        max_width=0
        max_height=0
		# ensure that at least one contour was found
        if len(cnts) > 0:
		# sort the contours according to their size in
		# descending order
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            
			# loop over the sorted contours
            rectangles=[]
            for c in cnts:
			# approximate the contour
                (x, y, w, h) = cv2.boundingRect(c)
                print (x, y, w, h)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
				# if our approximated contour has four points,
				# then we can assume we have found the paper
                if len(approx) == 4 and (h<500 or w <1000):
                    continue 
                if len(approx) == 4 :
					max_width=w
					max_height=h
					docCnt = approx
					rectangles.append(docCnt)
					break

        for rect in rectangles:
			paper = four_point_transform(img, docCnt.reshape(4, 2))
			warped = four_point_transform(img, docCnt.reshape(4, 2))

			threshold_img = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
			
			cnts = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			cnts = cnts[0] if imutils.is_cv2() else cnts[1]
			questionCnts = []
 
# loop over the contours
        for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
            if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
                questionCnts.append(c)


        # local threshold
		#  threshold_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
		#  blurred_img = cv2.GaussianBlur(threshold_img,(7,7),0)

        # print the used threshold value (if pixel < ret: pixel is set to 0 else: 255)
		#  inverted_img = cv2.bitwise_not(threshold_img)
        # Apply Canny edge detection
		# canny = cv2.Canny(img, 50, 240)
        #Adding Canny and Threshold value
		# adding = cv2.add(canny,inverted_img)
        #print the width and the lenght
		# print inverted_img.shape
        #crop the picture 
		# cropped = inverted_img[720:1450, 100:1240]
        #blur = cv2.GaussianBlur(adding,(5,5),0)
        # save the images
        result_filename = join(train_result_path, f)
        cv2.imwrite(result_filename, threshold_img)

        # display images
        # cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
        # cv2.imshow('image', inverted_img)

    # remove the break if you are not displaying the image
    # break
cv2.waitKey(0)
cv2.destroyAllWindows()