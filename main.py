from os import listdir
from os.path import isfile, join
import cv2

train_images_path = '../train'
test_images_path = '../test'

# loop through all of the files in the directory
for f in listdir(train_images_path):
    # the total path of the current file name
    filename = join(train_images_path, f)
    # check if it's a file
    if isfile(filename):
        # read the image file
        img = cv2.imread(filename, 0)
        # THRESH_OTSU determines the best threshold value based on the image histogram then uses this threshold
        ret, threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # print the used threshold value (if pixel < ret: pixel is set to 0 else: 255)
        print ret
        # display images
        cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
        cv2.imshow('image', threshold_img)

    # remove the break if you are not displaying the image
    break
cv2.waitKey(0)
cv2.destroyAllWindows()
