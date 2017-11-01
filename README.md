# dress-code-cnn

This piece of code uses hog(histogram-of-features) for person identification in an image. If there are more than one person in an image they are shown with a bounding box of green color. After person detection , the cropped images of persons are fed one by one into a cnn(convolutional neural network) built on keras(deep learning library) and then the dress code is appropriately predicted by the cnn as well as the gender.

Dependencies

Keras 2.0.8

Tensorflow(Keras runs on top of tensorflow)

Numpy

OpenCV 3.3


How to run


1. Save the image under test_image directory by image.jpg

2. Run mainprog.py(After execution the cropped images will appear in the main directory) so for retesting delete those images first)
