import cv2
import os

def draw_bounding_box(filename,width,height,label,xmin,ymin,xmax,ymax,save_path):
    
    # get the file path 
    images = "images"
    
    # Image Path
    img_path = os.path.join(os.getcwd(),images,label,filename)

    # Reading the Image
    im = cv2.imread(img_path)
    cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,255,0),2)
    cv2.imwrite(save_path, im)

    # for i in range(0, len(contours)):
    #     if (i % 2 == 0):
    #         cnt = contours[i]
    #         #mask = np.zeros(im2.shape,np.uint8)
    #         #cv2.drawContours(mask,[cnt],0,255,-1)
    #         x,y,w,h = cv2.boundingRect(cnt)
    cv2.waitKey()  
    cv2.destroyAllWindows()   
            
        
