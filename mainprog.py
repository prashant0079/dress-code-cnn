#!/usr/bin/env python

'''
example to detect upright people in images using HOG features and using cnn 
to predict dresscodes

Usage:
    save any image under test_image directory and name it as image.jpg

Press enter key to continue
'''

# Python 2/3 compatibility
from __future__ import print_function
import glob
import numpy as np
import cv2
from keras.models import model_from_json
from keras.preprocessing import image


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    d = 0
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
        image = img[y+pad_h:y+h-pad_h, x+pad_w:x+w-pad_w]
        filename = "%d.jpg"%d
        cv2.imwrite(filename, image)
        d = d + 1

if __name__ == '__main__':
    import sys
    from glob import glob
    import itertools as it

    print(__doc__)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    default = ['test_image/image.jpg'] if len(sys.argv[1:]) == 0 else []

    for fn in it.chain(*map(glob, default + sys.argv[1:])):
        print(fn, ' - ',)
        try:
            img = cv2.imread(fn)
            if img is None:
                print('Failed to load image file:', fn)
                continue
        except:
            print('loading error')
            continue

        found, w = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        draw_detections(img, found)
        draw_detections(img, found_filtered, 3)
        print('%d (%d) found' % (len(found_filtered), len(found)))
        cv2.imshow('img', img)
        ch = cv2.waitKey()
        if ch == 27:
            break
    cv2.destroyAllWindows()
    
#json_file1 = open('classifier1.json', 'r')
#loaded_model_json1 = json_file1.read()
#json_file1.close()
#loaded_model1 = model_from_json(loaded_model_json1)
# load weights into new model
#loaded_model1.load_weights("classifier1.h5")
#print("Loaded model from disk")
    

json_file2 = open('classifier2.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
loaded_model2 = model_from_json(loaded_model_json2)
# load weights into new model
loaded_model2.load_weights("classifier2.h5")
print("Loaded model from disk")

import glob
collection = glob.glob('*.jpg')
for c in collection:
    var = cv2.imread(c)
    cv2.imshow('image', var)
    cv2.waitKey(0)
    test_image = image.load_img(c, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #result1 = loaded_model1.predict(test_image)
    result2 = loaded_model2.predict(test_image)
    if(result2[0][0]==1):
        print('bikini-female')
    elif(result2[0][1]==1):
        print('burqa-female')    
    elif(result2[0][2]==1):
        print('coatpant-male')    
    elif(result2[0][3]==1):
        print('jeanstshirt-male')
    elif(result2[0][4]==1):
        print('kurtapyjama-male')    
    elif(result2[0][5]==1):
        print('skirts-female') 
    cv2.waitKey(0)    

cv2.destroyAllWindows()
    
    
    
    