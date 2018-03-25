import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def visualizationPCA_LDA():
    with open('/image_cv1_test.pkl', 'rb') as f: 
        x_test_image, Y = pickle.load(f)
        
    with open('/text_cv1_test.pkl', 'rb') as f:  
        x_test_text = pickle.load(f)


    X = []
    for i in range(0,len(x_test_image)):
        joint = np.concatenate([x_test_image[i],x_test_text[i]])
        X = np.concatenate([X,joint])

    X= X.reshape(len(x_test_image),2816)
    y = np.zeros(len(Y))
    for i in range (0,len(Y)):
        y[i] = np.argmax(Y[i])


    target_names = ['Infrastructure','Nature', 'Fires' , 'Flood' , 'Human', 'Non-damage']


    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange','black','red','green']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2,3,4,5], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of disaster dataset')

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2,3,4,5], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of disaster dataset')

    plt.show()
