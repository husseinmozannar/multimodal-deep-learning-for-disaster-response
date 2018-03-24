import os
import shutil
import random
#used to create "hard" different folds using 4-fold or Monte Carlo of dataset images and text

def copy_dir(src, dst,follow_sym=True):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    if os.path.isdir(src):
        shutil.copyfile(src, dst, follow_symlinks=follow_sym)
        shutil.copystat(src, dst, follow_symlinks=follow_sym)
    return dst

def createCVset(parentDir, dataset):
    for i in range (1,6): # create sets
        dirName = parentDir+'/'+'CV'+str(i)
        shutil.copytree(parentDir+'/'+dataset,dirName+'-train')
        shutil.copytree(parentDir+'/'+dataset,dirName+'-test',copy_function=copy_dir)
        shutil.copytree(parentDir+'/'+dataset,dirName+'-val',copy_function=copy_dir)


    dirName = parentDir+'/'+'CV'+str(5)
    for subdir, dirs , files in os.walk(dirName+'-train'): #distribute test/val
        copied=0
        print (subdir)
        for file in files:
            if not subdir.endswith('images'):
                break
            
            pick_file = random.randint(1,100)
            pick_test = random.randint(1,25)
            if pick_file <=25:
                copied+=1
                if pick_test <=18:
                    destination = subdir.replace('train','test')
                    destination = destination.replace('CV5','CV1')
                    shutil.move(os.path.join(subdir,file),destination)
                    
                    destination = destination.replace('images','text')
                    textFile = file[:-4] +'.txt'
                    textsubdir = subdir.replace('images','text')
                    try:
                        shutil.move(os.path.join(textsubdir,textFile),destination)
                    except:
                        print("some error occured")
                else:
                    destination = subdir.replace('train','val')
                    destination = destination.replace('CV5','CV1')
                    shutil.move(os.path.join(subdir,file),destination)
                    
                    destination = destination.replace('images','text')
                    textFile = file[:-4] +'.txt'
                    textsubdir = subdir.replace('images','text')
                    try:
                        shutil.move(os.path.join(textsubdir,textFile),destination)
                    except:
                        print("some error occured")
            elif pick_file <=50:
                if pick_test <=18:
                    destination = subdir.replace('train','test')
                    destination = destination.replace('CV5','CV2')
                    shutil.move(os.path.join(subdir,file),destination)
                    
                    destination = destination.replace('images','text')
                    textFile = file[:-4] +'.txt'
                    textsubdir = subdir.replace('images','text')
                    try:
                        shutil.move(os.path.join(textsubdir,textFile),destination)
                    except:
                        print("some error occured")
                else:
                    destination = subdir.replace('train','val')
                    destination = destination.replace('CV5','CV2')
                    shutil.move(os.path.join(subdir,file),destination)
                    
                    destination = destination.replace('images','text')
                    textFile = file[:-4] +'.txt'
                    textsubdir = subdir.replace('images','text')
                    try:
                        shutil.move(os.path.join(textsubdir,textFile),destination)
                    except:
                        print("some error occured")
            elif pick_file <=75:
                if pick_test <=18:
                    destination = subdir.replace('train','test')
                    destination = destination.replace('CV5','CV3')
                    shutil.move(os.path.join(subdir,file),destination)
                    
                    destination = destination.replace('images','text')
                    textFile = file[:-4] +'.txt'
                    textsubdir = subdir.replace('images','text')
                    try:
                        shutil.move(os.path.join(textsubdir,textFile),destination)
                    except:
                        print("some error occured")
                else:
                    destination = subdir.replace('train','val')
                    destination = destination.replace('CV5','CV3')
                    shutil.move(os.path.join(subdir,file),destination)
                    
                    destination = destination.replace('images','text')
                    textFile = file[:-4] +'.txt'
                    textsubdir = subdir.replace('images','text')
                    try:
                        shutil.move(os.path.join(textsubdir,textFile),destination)
                    except:
                        print("some error occured")
            elif pick_file <=100:
                if pick_test <=18:
                    destination = subdir.replace('train','test')
                    destination = destination.replace('CV5','CV4')
                    shutil.move(os.path.join(subdir,file),destination)
                    
                    destination = destination.replace('images','text')
                    textFile = file[:-4] +'.txt'
                    textsubdir = subdir.replace('images','text')
                    try:
                        shutil.move(os.path.join(textsubdir,textFile),destination)
                    except:
                        print("some error occured")
                else:
                    destination = subdir.replace('train','val')
                    destination = destination.replace('CV5','CV4')
                    shutil.move(os.path.join(subdir,file),destination)
                    
                    destination = destination.replace('images','text')
                    textFile = file[:-4] +'.txt'
                    textsubdir = subdir.replace('images','text')
                    try:
                        shutil.move(os.path.join(textsubdir,textFile),destination)
                    except:
                        print("some error occured")

    for i in range(1,5):#now remove data from each train set that are in val/test
        dirName = parentDir+'/'+'CV'+str(i)
        for subdir, dirs , files in os.walk(dirName+'-val'):
            copied=0
            for file in files:
                destination = subdir.replace('val','train')
                os.remove(os.path.join(destination,file))
        dirName = parentDir+'/'+'CV'+str(i)
        for subdir, dirs , files in os.walk(dirName+'-test'):
            copied=0
            for file in files:
                destination = subdir.replace('test','train')
                os.remove(os.path.join(destination,file))




def createMCset(parentDir, dataset):
    for i in range(1,5):
        dirName = parentDir+'/'+'CV'+str(i)
        shutil.copytree(parentDir+'/'+dataset,dirName+'-train')
        shutil.copytree(parentDir+'/'+dataset,dirName+'-val',copy_function=copy_dir)
        for subdir, dirs , files in os.walk(dirName+'-train'):
            copied=0
            for file in files:
                pick_file = random.randint(1,10)
                if copied==0:
                    destination = subdir.replace('train','val')
                    shutil.move(os.path.join(subdir,file),destination)
                    copied+=1
                elif pick_file <=2:
                    copied+=1
                    destination = subdir.replace('train','val')
                    shutil.move(os.path.join(subdir,file),destination)


