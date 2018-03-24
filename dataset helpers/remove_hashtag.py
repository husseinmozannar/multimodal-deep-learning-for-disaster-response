import fileinput
import shutil
import os


#text preprocessing
def removeHashtags(source):
    files = os.listdir(source)
    for file in files:
        if file.endswith(".txt"):
            # Read in the file
            with open(source+'/'+file, 'r') as textfile :
                filedata = textfile.read()
            # Replace the target string
            filedata = filedata.replace('#', ' ')
            # Write the file out again
            with open(source+'/'+file, 'w') as textfile:
                textfile.write(filedata)
