from os import listdir, makedirs
from os.path import isfile, join, exists
import imutils
import cv2
from imutils.perspective import four_point_transform
import matplotlib.pyplot as plt


def question_number(y_value, column):
    row = 1
    while y_value > 30:
        y_value -= 40
        row += 1
    return row + (15 * column)


def answer(x_value):
    if x_value < 50:
        return 'A'
    elif x_value < 90:
        return 'B'
    elif x_value < 130:
        return 'C'
    else:
        return 'D'


correct_answers = {1:  'B', 2:  'C', 3:  'A', 4:  'A', 5:  'D', 6:  'A', 7:  'C', 8:  'C', 9:  'A', 10: 'C', 11: 'A',
                   12: 'B', 13: 'C', 14: 'C', 15: 'B', 16: 'A', 17: 'D', 18: 'B', 19: 'C', 20: 'B', 21: 'D', 22: 'C',
                   23: 'D', 24: 'B', 25: 'D', 26: 'C', 27: 'D', 28: 'D', 29: 'B', 30: 'C', 31: 'B', 32: 'B', 33: 'D',
                   34: 'C', 35: 'B', 36: 'C', 37: 'B', 38: 'C', 39: 'C', 40: 'A', 41: 'B', 42: 'B', 43: 'C', 44: 'C',
                   45: 'B'}


train_images_path = '../train'
test_images_path = '../test'

train_result_path = '../train_result'
cropped_path = '../cropped'

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
        # ret, threshold_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
                # print (x, y, w, h)
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

                questions_cols = [erosion[70:665, 90:270],
                                  erosion[70:665, 418:598],
                                  erosion[70:665, 745:925]]

                answers = {}
                score = 0
                for i in range(len(questions_cols)):
                    circles = cv2.HoughCircles(questions_cols[i], cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10,
                                               minRadius=2, maxRadius=20)
                    if circles is not None:
                        for x, y, r in circles[0]:
                            question = question_number(y, i)
                            # if a question appeared more than once remove it's mark
                            # (2 choices are picked for the same question)
                            # print x, y, r
                            if question not in answers:
                                answers[question] = answer(x)
                                if answers[question] == correct_answers[question]:
                                    score += 1
                            else:
                                score -= 1
                            cv2.circle(backtorgb, (int(x + 90 + 330*i), int(y) + 70), r, (0, 0, 255), -1)
                # fig, ax = plt.subplots()
                # im = ax.imshow(backtorgb[70:665, :], interpolation='none')
                # plt.show()
                # cv2.imshow('image',backtorgb)
                # save the images
                print answers
                print score
                result_filename = join(train_result_path, f)
                cv2.imwrite(result_filename, backtorgb)

cv2.waitKey(0)
