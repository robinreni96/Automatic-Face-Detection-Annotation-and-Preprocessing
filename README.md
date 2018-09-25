# Automatic-Face-Detection-Annotation-and-Preprocessing
For creating a facial recognition model we need the facial landmarked data from the images . To get it , we need to manually label all the images using the labeltools , annotate the image with their coordiantes and then convert it to a csv file . Then we preprocess it to respective data file format like tfrecord etc. To make it easy for the AI Developers , I coded this module which can automatically detect , annotate , collect the coordinates , convert to csv and to tfrecord . And I added a feature to visulaize your detected face on the image according to the classes also.

![alt text](https://github.com/robinreni96/Automatic-Face-Detection-Annotation-and-Preprocessing/tree/master/resource/18.jpg)

