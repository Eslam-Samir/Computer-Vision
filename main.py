import re
import math
import csv
import cv2
import imutils
import numpy as np
from imutils import contours
from os import listdir, makedirs
from os.path import isfile, join, exists
from data import testdata, traindata


def question_number(row, column):
    return row + (15 * column)


correct_answers = {1:  'B', 2:  'C', 3:  'A', 4:  'A', 5:  'D', 6:  'A', 7:  'C', 8:  'C', 9:  'A', 10: 'C', 11: 'A',
                   12: 'B', 13: 'C', 14: 'C', 15: 'B', 16: 'A', 17: 'D', 18: 'B', 19: 'C', 20: 'B', 21: 'D', 22: 'C',
                   23: 'D', 24: 'B', 25: 'D', 26: 'C', 27: 'D', 28: 'D', 29: 'B', 30: 'C', 31: 'B', 32: 'B', 33: 'D',
                   34: 'C', 35: 'B', 36: 'C', 37: 'B', 38: 'C', 39: 'C', 40: 'A', 41: 'B', 42: 'B', 43: 'C', 44: 'C',
                   45: 'B'}

options = ['A', 'B', 'C', 'D']

threshold = 190

train_images_path = '../train1'
train_result_path = '../train_result1'

if not exists(train_result_path):
    makedirs(train_result_path)

data = [['FileName', 'Mark']]

lst = listdir(train_images_path)

wrong = {}

# loop through all of the files in the directory
for f in lst:
    # the total path of the current file name
    filename = join(train_images_path, f)
    # check if it's a file
    if isfile(filename):
        print filename
      
        # read the image file
        img = cv2.imread(filename, 0)
        RGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cs = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=10,
                              minRadius=40, maxRadius=50)

        xs = []
        ys = []
        if cs is not None:
            for x, y, r in cs[0]:
                if y > 1400:
                    # print x,y,r
                    cv2.circle(RGB, (x, y), r, (255, 0, 0), -1)
                    xs.append(x)
                    ys.append(y)

        result_filename = join(train_result_path, "step1_Hough_" + f)
        cv2.imwrite(result_filename, RGB)

        y_diff = ys[0] - ys[1]
        x_diff = xs[0] - xs[1]
        theta = math.atan(y_diff / x_diff) * 180 / math.pi

        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
        rotated_image = cv2.warpAffine(img, M, (cols, rows))

        cs = cv2.HoughCircles(rotated_image, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=10,
                              minRadius=40, maxRadius=50)
        RGB = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2RGB)

        xs = []
        ys = []
        if cs is not None:
            for x, y, r in cs[0]:
                if y > 1400:
                    # print x,y,r
                    cv2.circle(RGB, (x, y), r, (255, 0, 0), -1)
                    xs.append(x)
                    ys.append(y)

        result_filename = join(train_result_path, "step2_rotated_" + f)
        cv2.imwrite(result_filename, RGB)

        if xs[0] > xs[1]:
            xs[0], xs[1] = xs[1], xs[0]
            ys[0], ys[1] = ys[1], ys[0]

        x1 = xs[0] - 20
        x2 = xs[1] + 80
        y1 = ys[0] - 780
        y2 = ys[0] - 150

        cropped = rotated_image[y1:y2, x1:x2]

        result_filename = join(train_result_path, "step3_cropped_rotated_" + f)
        cv2.imwrite(result_filename, cropped)

        rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)

        # apply gaussian blur to remove random dots
        blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        questions_cols = [thresh[:, 0:180],
                          thresh[:, 328:508],
                          thresh[:, 656:836]]

        score = 0
        answers = {}
        for i in range(len(questions_cols)):
            result_filename = join(train_result_path, "Col" + str(i) + "_" + f)
            cv2.imwrite(result_filename, questions_cols[i])
            cnts = cv2.findContours(questions_cols[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            questionCnts = []

            # loop over the contours
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                circle_ratio = w / float(h)
                if w >= 18 and h >= 18 and 0.4 <= circle_ratio <= 1.9:
                    questionCnts.append(c)

            if len(questionCnts) > 60:
                j = 0
                while j < len(questionCnts):
                    (x, y, w, h) = cv2.boundingRect(questionCnts[j])
                    circle_ratio = w / float(h)
                    if not (w >= 20 and h >= 20 and (0.9 <= circle_ratio <= 1.2)):
                        del questionCnts[j]
                    else:
                        j += 1
            cv2.drawContours(rgb[:, i * 328:180 + i * 328], questionCnts, -1, (255, 0, 0), 1)
            questionCnts = list(contours.sort_contours(questionCnts, method="top-to-bottom")[0])

            result_filename = join(train_result_path, f)
            cv2.imwrite(result_filename, rgb)

            number = 1
            j = 0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

            while j < len(questionCnts):
                n = 1
                (x_val, y_val, w_val, h_val) = cv2.boundingRect(questionCnts[j])
                for q in questionCnts[j+1:]:
                    (x, y, w, h) = cv2.boundingRect(q)
                    if abs(y - y_val) > 20:
                        break
                    n += 1

                choices = contours.sort_contours(questionCnts[j:j+n])[0]
                (x_val, y_val, w_val, h_val) = cv2.boundingRect(choices[0])
                bubbled = None
                multi_bubbles = False
                question_no = question_number(number, i)
                x_prev = x_val

                m = 0
                for n in range(4):
                    (x, y, w, h) = cv2.boundingRect(choices[m])
                    mask = np.zeros(questions_cols[i].shape, dtype="uint8")
                    if n == 0 and x_val > 40:
                        cv2.rectangle(mask, (x_prev - 40, y_val - 2), (x_prev - 15, y_val + 22), 255, -1)
                    elif n >= len(choices) or x - x_prev > 60:
                        cv2.rectangle(mask, (x_prev + 40, y_val - 2), (x_prev + 65, y_val + 22), 255, -1)
                        x_prev += 40
                    else:
                        cv2.drawContours(mask, [choices[m]], -1, 255, -1)
                        x_prev = x
                        m += 1
                        if m == len(choices):
                            m -= 1

                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.bitwise_and(questions_cols[i], questions_cols[i], mask=mask)
                    total = cv2.countNonZero(mask)

                    # if question_no == 1:
                    result_filename = join(train_result_path, "Q" + str(question_no) + "C" + str(n) + "_" + f)
                    cv2.imwrite(result_filename, mask)

                    if bubbled is not None and 270 < bubbled[0] and 270 < total:
                        multi_bubbles = True
                    if bubbled is None or total > bubbled[0]:
                        bubbled = (total, options[n])

                if bubbled[0] > threshold and multi_bubbles is False:
                    answers[question_no] = bubbled[1]
                    if bubbled[1] == correct_answers[question_no]:
                        score += 1
                j += len(choices)
                number += 1

        print answers
        print score
        data.append([f, score])
        if f in traindata and score != traindata[f]:
            wrong[f] = traindata[f] - score
        elif f in testdata and score != testdata[f]:
            wrong[f] = testdata[f] - score
print wrong
print len(wrong)

with open('sub.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(data)
