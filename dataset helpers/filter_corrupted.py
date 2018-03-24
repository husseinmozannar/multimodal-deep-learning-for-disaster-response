import cv2
import os
directory = ""

#used to filter corrupted or images in bad format
def filterBadFormat():
    i=0
    for subdir,dirs, files in os.walk(directory):
        for file in files:
            try:
                image = cv2.imread(subdir+'/'+file)
                if len(image.shape)>3:
                    print("image shape irregular")
                if image.shape[0]<20 or image.shape[1]<20:
                    print("very small imaage shape")
            except:
                print(subdir+'/'+file)
                
                os.remove(subdir+'/'+file)
            i+=1
            if i%100==0:
                print("processed till now "+str(i))
        
