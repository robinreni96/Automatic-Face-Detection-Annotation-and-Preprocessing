import os 
import sys
import argparse
from bounding import draw_bounding_box


def read_csv(csv_name):
    # reading the file 
    file = open(csv_name, 'r')
    file_data = file.read()

    # Split lines into list.
    file_data_lines = file_data.split('\n')
    file_data_lines

    # Create the final cleaned list.
    cleaned_file = []
    # Loop to iterate and process each line.
    for line in file_data_lines:
        processed_line = line.split(',')
        cleaned_file.append(processed_line)

    return cleaned_file

def create_output_dirs(class_label):
    # defining the output directory
    output_path=os.path.join(os.getcwd(),"output")
    # defining the output label directory
    for i in class_label:
        # saving directory
        save_path=os.path.join(output_path,i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    return output_path



def export(csv_name,class_label):
    
    # read the csv and return the list
    data_list = read_csv(csv_name)

    # setting up the output directory
    primary_path = create_output_dirs(class_label)

    # Drawing boundary box for the images
    for i in range(1,len(data_list)):
        if len(data_list[i]) == 8 :
            filename=data_list[i][0]
            width=int(float(data_list[i][1]))
            height=int(float(data_list[i][2]))
            label=data_list[i][3]
            xmin=int(float(data_list[i][4]))
            ymin=int(float(data_list[i][5]))
            xmax=int(float(data_list[i][6]))
            ymax=int(float(data_list[i][7]))
        
        save_path = os.path.join(primary_path,label,filename)
        draw_bounding_box(filename,width,height,label,xmin,ymin,xmax,ymax,save_path)

    print("Successfully Exported the Output")

