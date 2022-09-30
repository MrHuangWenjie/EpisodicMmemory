import cv2
import os
import numpy as np
# from agent import Visual
# vision = Visual()


# imgs = os.listdir('rep/')
# for img in imgs:
#     img_array = cv2.imread('imgs_annotated/' + img)
#     img_array = cv2.resize(img_array, (640, 480))
#     coordinates = vision.object_filter(img_array)
#     b = vision.salience_filter(coordinates, img_array)

imgs = os.listdir('rep_dete/')
cv2.namedWindow('img')
for img in imgs:
    label = img.split('_')[1]
    if label == 'toy':
        print(img)
        img_array = np.load('rep_dete/' + img)
        img_array = np.reshape(img_array, [64, 64, 5]).transpose([2, 0, 1])[:3]
        img_array = img_array.transpose([1, 2, 0])
        cv2.imshow('img', img_array)
        cv2.waitKey()
    # 223 377 327 247 116 186

