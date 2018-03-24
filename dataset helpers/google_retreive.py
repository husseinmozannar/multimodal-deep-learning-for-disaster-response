from imagesoup import ImageSoup
import os
import shutil
import glob

directory= ""
keywords = { 'injured people' }

#used to collect images using Google Search with a list of kywords
def collectImagesGoogle():
    soup = ImageSoup()
    for key in keywords:
        print("collecting images for keyword "+key)
        imagedir = directory+'/'+key
        if not os.path.exists(imagedir):
            os.makedirs(imagedir)
        images = soup.search(key,image_size='medium', n_images=1000)
        for i in range(0,len(images)):
            image = images[i]
            url = image.URL
            url = url.split('/')
            name = url[len(url)-1]
            name = key + str(i) 
            name = name +'.jpg'
            try:
                image.to_file(imagedir+'/'+name)
            except:
                print("some error occured")
            if i%100==0:
                print("collected so far "+ str(i))
            
            
collectImages()
