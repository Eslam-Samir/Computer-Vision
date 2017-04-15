import re
import math
import csv
import imutils
import cv2
import matplotlib.pyplot as plt
from os import listdir, makedirs
from os.path import isfile, join, exists
from imutils.perspective import four_point_transform


def question_number(y_value, column):
    row = 1
    while y_value > 45:
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


def convert(text):
    if text.isdigit():
        return int(text)
    else:
        return text.lower()


def natural_sort(l):
    return sorted(l, key=(lambda key: [convert(c) for c in re.split('([0-9]+)', key)]))


correct_answers = {1:  'B', 2:  'C', 3:  'A', 4:  'A', 5:  'D', 6:  'A', 7:  'C', 8:  'C', 9:  'A', 10: 'C', 11: 'A',
                   12: 'B', 13: 'C', 14: 'C', 15: 'B', 16: 'A', 17: 'D', 18: 'B', 19: 'C', 20: 'B', 21: 'D', 22: 'C',
                   23: 'D', 24: 'B', 25: 'D', 26: 'C', 27: 'D', 28: 'D', 29: 'B', 30: 'C', 31: 'B', 32: 'B', 33: 'D',
                   34: 'C', 35: 'B', 36: 'C', 37: 'B', 38: 'C', 39: 'C', 40: 'A', 41: 'B', 42: 'B', 43: 'C', 44: 'C',
                   45: 'B'}


train_images_path = '../train'
test_images_path = '../test'
train_result_path = '../train_result'

if not exists(train_result_path):
    makedirs(train_result_path)

data = [['FileName', 'Mark']]

lst = listdir(train_images_path)
lst = natural_sort(lst)
# loop through all of the files in the directory
for f in lst:
    # the total path of the current file name
    filename = join(train_images_path, f)
    # check if it's a file
    if isfile(filename):
        print filename
      
        # read the image file
        img = cv2.imread(filename, 0)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # apply gaussian blur to remove random dots
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edged = cv2.Canny(blurred, 130, 280)
        cs = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=10,
                                   minRadius=40, maxRadius=50)
        xs = []
        ys = []
        if cs is not None:
            for x, y, r in cs[0]:
                if y > 1400:
                    xs.append(x)
                    ys.append(y)

        y_diff = ys[0] - ys[1]
        x_diff = xs[0] - xs[1]
        theta = math.atan(y_diff / x_diff) * 180 / math.pi

        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
        rotated_edged = cv2.warpAffine(edged, M, (cols, rows))
        rotated_image = cv2.warpAffine(rgb, M, (cols, rows))

        cs = cv2.HoughCircles(rotated_edged, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=10,
                              minRadius=40, maxRadius=50)
        xs = []
        ys = []
        if cs is not None:
            for x, y, r in cs[0]:
                if y > 1400:
                    xs.append(x)
                    ys.append(y)

        if xs[0] > xs[1]:
            xs[0], xs[1] = xs[1], xs[0]
            ys[0], ys[1] = ys[1], ys[0]

        x1 = xs[0] - 110
        x2 = xs[1] + 110
        y1 = ys[0] - 820
        y2 = ys[0] - 120

        cropped = rotated_image[y1:y2, x1:x2]
        cropped_edged = rotated_edged[y1:y2, x1:x2]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        closing = cv2.morphologyEx(cropped_edged, cv2.MORPH_CLOSE, kernel)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion = cv2.erode(closing, kernel2, iterations=1)

        questions_cols = [erosion[50:665, 90:270],
                          erosion[50:665, 418:598],
                          erosion[50:665, 745:925]]

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
                    cv2.circle(cropped, (int(x + 90 + 330*i), int(y) + 50), r, (0, 0, 255), -1)
        #fig, ax = plt.subplots()
        #im = ax.imshow(cropped[50:665, 745:925], interpolation='none')
        #plt.show()
        # cv2.imshow('image', paper)
        # save the images
        print answers
        print score
        data.append([f, score])
        result_filename = join(train_result_path, f)
        cv2.imwrite(result_filename, cropped)

with open('submission.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(data)
