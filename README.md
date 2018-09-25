# Automatic-Face-Detection-Annotation-and-Preprocessing [![Build Status][travis-image]][travis]

[travis-image]: https://travis-ci.org/robinreni96/Automatic-Face-Detection-Annotation-and-Preprocessing.svg?branch=master
[travis]: https://travis-ci.org/robinreni96/Automatic-Face-Detection-Annotation-and-Preprocessing

For creating a facial recognition model we need the facial landmarked data from the images . To get it , we need to manually label all the images using the labeltools , annotate the image with their coordiantes and then convert it to a csv file . Then we preprocess it to respective data file format like tfrecord etc. To make it easy for the AI Developers , I coded this module which can automatically detect , annotate , collect the coordinates , convert to csv and to tfrecord . And I added a feature to visulaize your detected face on the image according by their respective classes.


<p align="center"> 
<img src="https://github.com/robinreni96/Automatic-Face-Detection-Annotation-and-Preprocessing/blob/master/resource/18.jpg">
</p>

## Compatibility
The code is tested and developed  in ubuntu 18.04 and using pyton 3.6.But the code has the realiability to run on most of the configuration . If you face issues , do open up an issue for this repo .All the package dependencies are mentioned in requirements.txt.

## Workings
1. Preprocessing all the images to a standard size and format
2. Loading the preprocessed image
3. Detecting the Face in the image using MTCNN or Harr-Cascade Algorithm and removing bad images
4. Getting the face coordinates
5. Writing into csv
6. Converting into tfrecord
7. Exporting the images with the face bounding box for debuging

**Core Functionality**
+ `main.py` - Parse the arguments , load the images , call the detect,cordinates,preprocess functions
+ `coordinates.py` - Using MTCNN it detects the facial boundary coordinates
+ `cascade.py` - Using Harr-Cascade detects the facial boundary coordinates
+ `generate_tfrecord.py` - Used to generate the tfrecord for the csv
+ `dataset_util` - some utility functions for generate_tfrecord
+ `check.py` - load the processed image and export the output with the bounding box
+ `bounding.py` - draw the bounding box using opencv
+ `requirements.txt` - contains all the packages and their versions

<p align="center"> 
<img src="https://github.com/robinreni96/Automatic-Face-Detection-Annotation-and-Preprocessing/blob/master/resource/structue.png">
</p>

### Note : The images should be in jpg format and each image of a class should have only one person image.

## Steps to run the code for your custom dataset
1. Create a python virtual environment and pip install the requirements.txt
2. Then prepare the images folder according to structure
3. Create a empty output directory
4. To detect the face using Facenet run `python main.py images_directory_path csv_name tfrecord_name facenet`.Example : `python main.py /home/robinreni/Document
s/Automatic-Face-Detection-Annotation-and-Preprocessing-master/images train.csv train.record facenet`
5. To detect the face using Harr-Cascade run `python main.py images_directory_path csv_name tfrecord_name harr`. Example : `python main.py /home/robinreni/Document
s/Automatic-Face-Detection-Annotation-and-Preprocessing-master/images train.csv train.record harr`
6. After code execution , you will get a csv and tfrecord file . To view the detection is perfect , go to the output directory you can view all the images with detection according to their respective class.
7. Simple !

### Additional Feature 
It automatically removes the all the bad format images and multiple face images

## For More Reference :
+ MTCNN : https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf
+ Harr-Cascade : https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html
+ To know more about tfrecord : https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/

ALL THE WORKS ARE DEVELOPED WITH THE GUIDANCE AND SUPPORT FROM [INNOVATION INCUBATOR](https://innovationincubator.com/)




