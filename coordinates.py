from scipy import misc
import tensorflow as tf
import numpy as np
import os
import align.detect_face




def embeddings(image_path,image_size,margin,gpu_memory_fraction):

    minsize = 10 # minimum size of face
    threshold = [0.6,0.7,0.7] # p,r,o nets threshold
    factor = 0.709 # Standard Scaling Factor
    x = y = w = h = 0

    with tf.Graph().as_default():
        #creating a tf graph and also setting the gpu options
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        
        #Defining the session to run
        sess= tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
        with sess.as_default():
            # Structuring pnet , rnet and onet
            pnet , rnet , onet = align.detect_face.create_mtcnn(sess,None)

    # Reading the image using misc
    img=misc.imread(image_path)
    img = img[:,:,0:3]
    img_size = np.asarray(img.shape)[0:2]

    # detecting the face and getting the coordinates of the bounding box
    bounding_boxes , _ = align.detect_face.detect_face(img,minsize,pnet,rnet,onet,threshold,factor)
    nrof_faces = bounding_boxes.shape[0]

    if nrof_faces > 0:
        det = bounding_boxes[:,0:4]
        det_arr =[]
        det_arr.append(np.squeeze(det))

        for i , det in enumerate(det_arr):
            det = np.squeeze(det)

            if (len(det.shape)==1):
                bb = np.zeros(4,dtype=np.int32)
                bb[0] = x = np.maximum(det[0]-margin/2,0)
                bb[1] = y = np.maximum(det[1]-margin/2,0)
                bb[2] = w = np.minimum(det[2]+margin/2, img_size[1])
                bb[3] = h = np.minimum(det[3]+margin/2, img_size[0])
            
            else :
                x = y = w = h = 0

    # Calculating the coordinates
    xmin , ymin , xmax , ymax = x , y , w , h

    return xmin , ymin , xmax , ymax






