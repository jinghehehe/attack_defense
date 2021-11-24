
import os
import cv2 
for filename in os.listdir("/home/jf/detect/CCPD/images/defense_adv"):
    print(filename)
    image=cv2.imread("/home/jf/detect/CCPD/images/defense_adv/"+filename)
    #cv2.imshow("image", image) 
    size = image.shape
    res=cv2.resize(image,(800,800),interpolation=cv2.INTER_CUBIC)
    res=cv2.resize(res,(size[0],size[1]),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/home/jf/detect/CCPD/images/defense_adv/"+filename, res)
