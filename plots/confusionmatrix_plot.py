import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


# from https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
def plotConfusionMatrix():
    conf_arr = [[231.0 ,  6  , 2  , 5 ,  2  , 4],
     [ 18  ,75  , 1,   0 ,  0  , 5],
     [  0  , 0 , 46  , 0 ,  0 ,  1],
     [  5 ,  5  , 0  ,60   ,0  , 3],
     [  1  , 0  , 0 ,  1 , 42,   2],
     [  0   ,1  , 1 ,  0  , 0, 497]]

    array = conf_arr

    df_cm = pd.DataFrame(array, range(6),
                      range(6))
    #plt.figure(figsize = (10,7))
    sn.set(font_scale=1.2)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 18},fmt=".4g")# font size
    plt.savefig('confusion_matrix1.png', format='png')


    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width=6
    height = 6

    for x in range(0,width):
        for y in range(0,height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'AaBCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix2.png', format='png')
