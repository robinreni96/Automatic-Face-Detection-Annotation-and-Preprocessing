import sys 
import os 
import argparse
import csv
from PIL import Image
from coordinates import embeddings
from generate_tfrecord import generate_tf
from cascade import detect_face
import cv2
import numpy as np
from check import export


def writeCsvFile(file_name,data,*args,**kwargs):
    # Opening and writing the csv file
    with open(file_name,"w") as f:
        writer =csv.writer(f)
        writer.writerows(data)

    print ("****** Successfully CSV File Generated *************")


def standard_size(image_dir):
    for dirs in os.listdir(image_dir):
        dir_path=os.path.join(image_dir,dirs)
        for f in os.listdir(dir_path):
            img_p = os.path.join(image_dir,dirs,f)
            im = Image.open(img_p)
            image_data = np.asarray(im)
            file_name=f
            imResize = cv2.resize(image_data, (500, 500))
            try:
                im = Image.fromarray(imResize, 'RGB')
            except ValueError:
                os.remove(img_p)
                print("Bad Image Format, Removing")
                continue
            im.save(dir_path+'/'+f, 'JPEG', quality=90)


def image_process(image_dir,data_list,method):

    margin = 44 # Default Margin Value of the image
    counter = 0 # To count how many faces detected
    gpu_memory_fraction =1.0 # if we use GPU ,we define the upper bound of the GPU Memory
    standard_size(image_dir) # Converting all the image to standard size
    for dirs in os.listdir(image_dir):
        dir_path=os.path.join(image_dir,dirs)
        for f in os.listdir(dir_path):
            img_p = os.path.join(image_dir,dirs,f)
            label=img_p.split('/')[-2]
            file_name=f
            im = Image.open(img_p) # Opening the image using PILLOW
            w,h = im.size # getting the width and the height of the image
            size = im.size # for passing the face embeddings parameters
            if method == "harr":
                xmin,ymin,xmax,ymax=detect_face(im) # Using opencv method
            if method == "facenet":
                xmin,ymin,xmax,ymax = embeddings(img_p,size,margin,gpu_memory_fraction) # Calling the facenet embedding function
            
            if xmin == ymin == xmax == ymax == 0:
                # It will remove the undetected and error image
                os.remove(img_p)
                print("*"+img_p+"*")
                print("********** Error With the Image , So Removing  **********")
                
            else:
                # It will add the detected image
                counter += 1
                print("Face Detected and Processed : {}".format(counter))
                data_list.append([file_name,w,h,label,xmin,ymin,xmax,ymax]) # Appending in a list format
    
    print("**** Successfully image processed *********")

    return data_list


def parse_arguments(argv):
    # Defining the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir',type=str,help='Directory of them images')
    parser.add_argument('csv_name',type=str,help='Name of the csv')
    parser.add_argument('tfrecord_name',type=str,help='Name of the tfrecord')
    parser.add_argument('method',type=str,help='Method to use to detect Face')

    return parser.parse_args(argv)

# parsing the arguments
args=parse_arguments(sys.argv[1:])

# defining the list structure to convert to csv
data_list = [['filename','width','height','class','xmin','ymin','xmax','ymax']]

# process the images and getting the coordinate values as list
coordinate_list = image_process(args.image_dir,data_list,args.method)



# writing into csv file
writeCsvFile(args.csv_name, coordinate_list)

# conversion to tfrecord 
generate_tf(args.csv_name,args.tfrecord_name,args.image_dir)

# Exporting the output
class_labels=[]
for dirs in os.listdir(args.image_dir):
    class_labels.append(dirs)
export(args.csv_name,class_labels) # Calling the export function from check
