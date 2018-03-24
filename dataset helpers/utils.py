import shutil
import os
import glob

#various tools that are handy when building the dataset

sourceDir= ""
# change THIS with your parent directory, your parent directory should contain the folders
# where the images/captions.. reside for each hashtag in a seperate folder
# e.g : directory/#disaster

hashtags= {'#naturaldisaster','#earthquake','#floodwater','#destroyedbuilding',
           '#hurricanesandy','#wreckedcar','#disaster',
           '#buildingcollapse','#landslide','#destruction','#hurricaneharvey',
           '#warsyria','#explosion','#nature','#building','#like4like','#cars'}
# change THIS to your hashtags that you downloaded, remove others


labels= {"damaged_infrastructure","damaged_nature","fires","flood","human_damage","non_damage"}
#move image files to new directory
def moveText(source):
    imagedir=source+'/'+"text"
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    files = os.listdir(source)
    for file in files:
        if file.endswith(".txt") and not file.endswith("_location.txt") and "old" not in file:
            shutil.move(source+'/'+file, imagedir)

def moveImages(source):
    imagedir=source+'/'+"images"
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    files = os.listdir(source)
    for file in files:
        if file.endswith(".jpg"):
            shutil.move(source+'/'+file, imagedir)


def deleteALL(sourceDir):
    for label in labels:
        deleteFiles(sourceDir + '/'+label)


#delete files that dont match with the images
def deleteFiles(filedir):
    images = os.listdir(filedir +'/images')
    filedir = filedir + '/text'
    files = os.listdir(filedir)
    imageSet= set()
    for image in images:
        imageSet.add(image[:-4])#all the image name without the .jpg
        
    for file in files:
        if file.endswith("_location.txt") and file[:-13] not in imageSet: #_location.txt
            os.remove(filedir+'/'+file)
        elif file.endswith(".txt") and not file.endswith("_location.txt") and  file[:-4] not in imageSet: #.txt
            os.remove(filedir+'/'+file)
        elif file.endswith(".json") and file[:-14] not in imageSet: #_comments.json
            os.remove(filedir+'/'+file)

def changeNames(source, name):
    for subdir,dirs,files in os.walk(source):
        for file in files:
            newFileName = name +'_'+ file
            newFileName = subdir + '/' + newFileName
            oldFileName = subdir + '/' + file
            #print(oldFileName,newFileName)
            os.rename(oldFileName, newFileName)
            
def moveALL(filedir):
    oldsub =""
    for subdir,dirs, files in os.walk(directory):
        subDir = subdir.split('/')
        subDir = subDir[len(subDir)-1]
        subDir = subDir.split("\\")
        if len(subDir)>1:
            subDir = subDir[1]
            if subDir!= oldsub:
                imageDir= subdir
                name = subDir[1:]
                #print(imageDir,name)
                deleteFiles(imageDir)
            oldsub = subDir

def moveHashtags(filedir):
    for hashtag in hashtags:
        moveFiles(filedir+'/'+hashtag)
        
