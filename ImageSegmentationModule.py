import cv2
import numpy as np
import math
# from tensorflow import keras
import pathlib


# tf.compat.v1.enable_eager_execution()

# contrast = 1.5  # Contrast control (1.0-3.0)
# brightness = 0  # Brightness control (0-100)


# result = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


# Remove lines
def removeStructure(img, direction, v_size=3, h_size=9):
    # performs an 'opening' transformation (erosion followed by dilation) to the img to further remove noise)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    vertical_size = img.shape[0] // v_size  # should be //3. Change back after dataset mod
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical_mask = 255 - cv2.morphologyEx(img, cv2.MORPH_CLOSE, vertical_kernel)

    vertical_extract = cv2.morphologyEx(img, cv2.MORPH_CLOSE, vertical_kernel)

    just_vertical = cv2.add(img, vertical_mask)

    horizontal_size = img.shape[1] // h_size  # should be //9, Change back after dataset mod
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_mask = 255 - cv2.morphologyEx(img, cv2.MORPH_CLOSE, horizontal_kernel)
    just_horizontal = cv2.add(img, horizontal_mask)

    horizontal_extract = cv2.morphologyEx(img, cv2.MORPH_CLOSE, horizontal_kernel)

    if direction == 1:
        result = just_vertical
    else:
        result = just_horizontal

    # cv2.imshow("extracted features", vertical_extract)
    # cv2.waitKey(0)
    # cv2.destroyWindow("extracted features")

    # result = cv2.addWeighted(just_horizontal, 1, just_vertical, 0, 0) placeholder values
    return result


def getDirection(img, img_copy, threshold=1.5):
    # initial guess
    direction = 1
    img = removeStructure(img, direction)
    # Converting to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # applying Gaussian blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # experimenting with thresholding functions
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

    # print(img.shape[:3])
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 50, minLineLength=min(img.shape[:2]) / threshold, maxLineGap=5)
    gradients = []
    if lines is not None:
        # print(lines)
        for l in lines:
            for line in l:
                x1, y1, x2, y2 = line
                if x2 - x1 != 0:
                    gradients.append((y2 - y1) / (x2 - x1))
                else:
                    direction = 1
                    return direction
                cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
    mean_gradient = np.mean(gradients)
    if 2 > mean_gradient > -2:
        direction = 0
    else:
        direction = 1
    # cv2.imshow('detected lines', img_copy)
    # cv2.waitKey(0)
    return direction


def findContourBoxes(edged, im_copy, threshold=22):
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    counter = 1
    area = []
    coords = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        area.append(w * h)
    areaMean = np.mean(area)  # average area
    # areaSD = np.std(area)  # standard deviation of area
    # print("average area =", areaMean)
    # print("areaSD = ", areaSD)
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        area = w * h
        if area >= areaMean / threshold:  # / 20 and area <= areaMean*10):
            cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coords.append([x, y, w, h])
            # print(counter)
            counter += 1
    if not coords:
        return ['error']
    Coords = sorted(coords, key=lambda x: (x[0]))
    # print(Coords)

    return Coords


def getOverlap(box1, box2):
    return min([box2[0] + box2[2] - box1[0], box1[0] + box1[2] - box2[0]])


def drawGroupContours(Coords, img, threshold=5, groupAll=False):
    Coords = sorted(Coords, key=lambda x: (x[0]))
    if groupAll:
        min_x = Coords[0][0]
        min_y = sorted(Coords, key=lambda x: (x[1]))[0][1]
        groupedCoords = [[min_x,
                          min_y,
                          max([c[0] + c[2] for c in Coords]) - min_x,
                          max([c[1] + c[3] for c in Coords]) - min_y]]
    else:
        groupedCoords = [Coords[0]]
        Coords.remove(Coords[0])
        for i in range(0, len(Coords)):
            box = Coords[0]
            if (groupedCoords[-1][0] <= box[0] <= groupedCoords[-1][0] + groupedCoords[-1][2]) or (
                    groupedCoords[-1][0] <= box[0] + box[2] <= groupedCoords[-1][0] + groupedCoords[-1][2]):
                overlap = getOverlap(groupedCoords[-1], box)
                if (overlap >= groupedCoords[-1][2] / threshold or (
                        (groupedCoords[-1][0] <= box[0] <= groupedCoords[-1][0] +
                         groupedCoords[-1][2]) and (
                                groupedCoords[-1][0] <= box[0] + box[2] <=
                                groupedCoords[-1][0] + groupedCoords[-1][2]))):
                    groupedCoords[-1] = [min(groupedCoords[-1][0], box[0]), min(groupedCoords[-1][1], box[1]),
                                         max(groupedCoords[-1][2], box[0] + box[2] - groupedCoords[-1][0]),
                                         max(groupedCoords[-1][3], box[1] + box[3] - groupedCoords[-1][1],
                                             groupedCoords[-1][1] +
                                             groupedCoords[-1][3] - box[1])]
                else:
                    groupedCoords.append(box)
            else:
                groupedCoords.append(box)
            Coords.remove(box)
    for sortedBox in groupedCoords:
        cv2.rectangle(img, (sortedBox[0], sortedBox[1]), (sortedBox[0] + sortedBox[2], sortedBox[1] + sortedBox[3]),
                      (0, 0, 255), 2)

    return groupedCoords


def cropContourBoxes(coords, img):
    img_list = []
    areas = []
    aspect_ratios = []
    coords = sorted(coords, key=lambda x: (x[0]))
    for box in coords:
        symbol = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        areas.append(box[3] * box[2])
        aspect_ratios.append(box[2] / box[3])
        img_list.append(symbol)
    return img_list, areas, aspect_ratios


def resize(img_list, size):
    resized_imgs = []
    for img in img_list:
        # Get the aspect ratio of the input image
        height, width = img.shape[:2]
        if height > width:
            padded_img = cv2.copyMakeBorder(
                img,
                top=0,
                bottom=0,
                left=math.ceil((height - width) / 2),
                right=math.floor((height - width) / 2),
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        elif height < width:
            padded_img = cv2.copyMakeBorder(
                img,
                top=math.ceil((width - height) / 2),
                bottom=math.floor((width - height) / 2),
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        else:
            padded_img = img

        # Resize the image while preserving its aspect ratio
        resized_img = cv2.resize(padded_img, (int(size * 0.6), int(size * 0.6)))
        resized_img = cv2.copyMakeBorder(
            resized_img,
            top=math.ceil((size - int(size * 0.6)) / 2),
            bottom=math.floor((size - int(size * 0.6)) / 2),
            left=math.ceil((size - int(size * 0.6)) / 2),
            right=math.floor((size - int(size * 0.6)) / 2),
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        resized_img = cv2.dilate(resized_img, np.ones((2, 2), np.uint8))
        #note the below is experimental
        resized_img = cv2.resize(resized_img, (20,20))
        resized_img = cv2.resize(resized_img, (size, size))
        resized_imgs.append(resized_img)
    return resized_imgs


def segment(file, test=False, size = 50, Contour_thresh=22):
    img = cv2.imread(file)
    # img = file
    im_copy = img.copy()
    result = img.copy()
    initial = img.copy()

    if test:
        cv2.imshow("original image", img)
        cv2.waitKey(0)
        cv2.destroyWindow("original image")

    # checks the direction of the background lines by assuming a direction (1 being vertical)
    # and checks if there are lines remaining

    initial = removeStructure(initial, 1)
    direction = getDirection(initial, im_copy)

    # removes structure from original image given the direction
    result = removeStructure(result, direction)

    # Converting to greyscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # applying Gaussian blur
    blur = cv2.bilateralFilter(gray, 5, 40, 75)
    # blur = cv2.GaussianBlur(gray, (5,5), 0)

    # experimenting with thresholding functions
    # Do note that Otsu's binarization does not work well with empty backgrounds
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    if test:
        cv2.imshow("thresh1", thresh)
        cv2.waitKey(0)
        cv2.destroyWindow("thresh1")
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # applying Canny edge detection to extract contour lines
    edged = cv2.Canny(thresh, 70, 200)

    # returns a list of coordinates in [x, y, w, h] format of each contour box
    coords = findContourBoxes(edged, initial, Contour_thresh)

    # groups overlapping contours
    coords = drawGroupContours(coords, img)
    cv2.imwrite("byCapture.jpg", img)

    # stores the cropped images as a list, areas and aspect_ratio of each of the images to be used for
    # classification correction
    imglist, areas, aspect_ratios = cropContourBoxes(coords, thresh)

    if test:
        cv2.imshow("gray", gray)
        cv2.waitKey(0)
        cv2.destroyWindow("gray")

        cv2.imshow('blur', blur)
        cv2.waitKey(0)
        cv2.destroyWindow('blur')

        cv2.imshow("thresh", thresh)
        cv2.waitKey(0)
        cv2.destroyWindow('thresh')

        cv2.imshow('edged', edged)
        cv2.waitKey(0)
        cv2.destroyWindow('edged')

        cv2.imshow('contours', im_copy)
        cv2.waitKey(0)
        cv2.destroyWindow('contours')

        cv2.imshow('grouped contours', img)
        cv2.waitKey(0)
        cv2.destroyWindow('grouped contours')

        cv2.imshow('centroids', initial)
        cv2.waitKey(0)
        cv2.destroyWindow('centroids')

    resized_imgs = resize(imglist, size)
    centroids = []

    for coord in coords:
        x, y, w, h = coord
        centroids.append(((x + w) / 2, (y + h) / 2))
        cv2.circle(initial, (math.floor(x + w / 2), math.floor(y + h / 2)), 10, (255, 0, 0), 5)

    if test:
        i=0
        for img in imglist:
            print(img.shape)
        for img in resized_imgs:
            cv2.imshow(f'input[{i}]', img)
            cv2.waitKey(0)
            i+=1
    return resized_imgs, areas, aspect_ratios


def segmentDataset(img, test=False, groupAll=False):
    ''' Segementation function to reduce distribution mismatch between the training data and actual input data'''

    # as the image is inverted (white background and dark numbers),
    # an erosion has a dilation effect on the numbers/symbols as to simulate the effects of blurring
    # img = cv2.erode(img, np.ones((3, 3), np.uint8))
    im_copy = img
    initial = img
    initial = removeStructure(initial, 1, 1, 1) # only removes lines which cross the entire image, as images are
                                                # already cropped, the threshold needs to be decreased.
    direction = getDirection(initial, im_copy, threshold=1) # ensures only background lines are removed.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.erode(gray, np.ones((3, 3), np.uint8))
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    edged = cv2.Canny(thresh, 70, 200)
    coords = findContourBoxes(edged, im_copy, 100)
    if test:
        cv2.imshow('initial', initial)
        cv2.waitKey(0)
        print('direction : ', direction)
        cv2.imshow('result', img)
        cv2.waitKey(0)
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)
        cv2.imshow('edged', edged)
        cv2.waitKey(0)
        cv2.imshow('contours', im_copy)
        cv2.waitKey(0)
    if coords[0] == 'error':
        return ['error']
    coords = drawGroupContours(coords, initial, 100, groupAll) # groups all broken components if any.
    img_list, areas, aspect_ratios = cropContourBoxes(coords, thresh)
    resized_imgs = resize(img_list, 50)
    for img in resized_imgs:
        img = cv2.resize(img, (int(10), int(10)),  interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (500, 500),  interpolation=cv2.INTER_LINEAR)
    if test:
        cv2.imshow('grouped', initial)
        cv2.waitKey(0)
        for img in resized_imgs:
            cv2.imshow('input', img)
            cv2.waitKey(0)
    return resized_imgs


# segment('croppedinput.jpg', test=True, Contour_thresh=18)
# print(cv2.imread('Images/math4.jpg').shape)