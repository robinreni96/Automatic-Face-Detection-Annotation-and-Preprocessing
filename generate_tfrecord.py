import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
import dataset_util
from collections import namedtuple , OrderedDict


def class_img_dict(path):
    # Getting the class as a list
    class_list = os.listdir(path)

    class_dict ={}

    for i in range(0,len(class_list)):
        class_dict[class_list[i]] = i+1

    return class_dict



def split(df,group):
    # Spliting the object from the files
    data = namedtuple('data',['filename','label','object'])
    gb = df.groupby(group)
    li=[]
    for key, x in zip(gb.groups.keys(), gb.groups):
        d = data(key[0],key[1],gb.get_group(x))
        li.append(d)
    return li

def create_tf_example(group, path):
    # Class numeric labels as dict
    class_dict=class_img_dict(path)

    #Opening and readinf the files
    with tf.gfile.GFile(os.path.join(path,'{}/{}'.format(group.label,group.filename)),'rb') as fid:
        encoded_jpg = fid.read()

    # Encode the image in jpeg format to array values
    encoded_jpg_io= io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)

    # Setting up the image size
    width , height = image.size

    #Creating the boundary box coordinate instances such as xmin,ymin,xmax,ymax
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] /width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_dict[row['class']])

    # This is already exisiting code to convert csv to tfrecord
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def generate_tf(csv_name,tf_name,img_dir):
    #Creating a TFRecordWriter
    writer = tf.python_io.TFRecordWriter(tf_name)

    # selecting the path to the image folder
    path = os.path.join(os.getcwd(),'images')

    # Reading the csv from the data folder
    examples = pd.read_csv(csv_name)
    grouped = split(examples, ['filename','class'])
    for group in grouped:
        tf_example = create_tf_example(group,path)
        writer.write(tf_example.SerializeToString())

    writer.close()

    # After the conversion display the message
    output_path = os.path.join(os.getcwd(),tf_name)
    print('Successfully created the tfrecord for the images: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
