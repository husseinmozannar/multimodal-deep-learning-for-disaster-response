from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
# loading the iris dataset



  
def featuresClassifiers():
    with open('image_cv4_train.pkl', 'rb') as f: 
        x_train_image, Y_train = pickle.load(f)

    with open('image_cv4_test.pkl','rb') as f:  
        x_test_image, Y_test = pickle.load(f)

    with open('image_cv4_val.pkl','rb') as f: 
        x_val_image, Y_val = pickle.load(f)

    print("loaded images")
    with open('text_cv4_train.pkl', 'rb') as f:  
        x_train_text = pickle.load(f)

    with open('text_cv4_test.pkl','rb') as f:  
        x_test_text = pickle.load(f)

    with open('text_cv4_val.pkl','rb') as f:  
        x_val_text = pickle.load(f)

    #arrange data
    x_train = []
    x_test = []
    x_val = []

    for i in range(0,len(x_test_image)):
        joint = np.concatenate([x_test_image[i],x_test_text[i]])
        x_test = np.concatenate([x_test,joint])

    for i in range(0,len(x_train_image)):
        joint = np.concatenate([x_train_image[i],x_train_text[i]])
        x_train = np.concatenate([x_train,joint])
         
    for i in range(0,len(x_val_image)):
        joint = np.concatenate([x_val_image[i],x_val_text[i]])
        x_val = np.concatenate([x_val,joint])

    x_val = x_val.reshape(len(x_val_image),2816)
    x_test = x_test.reshape(len(x_test_image),2816)
    x_train = x_train.reshape(len(x_train_image),2816)

    # arrange y's for scikit learn format
    y_train = np.zeros(len(Y_train))
    y_test = np.zeros(len(Y_test))
    y_val = np.zeros(len(Y_val))

    for i in range (0,len(y_train)):
        y_train[i] = np.argmax(Y_train[i])
    for i in range (0,len(y_test)):
        y_test[i] = np.argmax(Y_test[i])  
    for i in range (0,len(y_val)):
        y_val[i] = np.argmax(Y_val[i])  



    # training an SVM classifier
    from sklearn.svm import SVC

    svm_model_linear = SVC(kernel = 'rbf', C = 1,max_iter=-1,probability=True,tol=1e-5,cache_size=400).fit(x_train, y_train)
    svm_predictions = svm_model_linear.predict(x_test)


    # model accuracy for X_test  
    accuracy = svm_model_linear.score(x_test, y_test)
    cr = classification_report(y_test, svm_predictions)
    # creating a confusion matrix
    print(cr)
    print(accuracy)

    svm_model_linear = SVC(kernel = 'linear', C = 1,max_iter=-1,probability=True,tol=1e-5,cache_size=400).fit(x_train, y_train)
    svm_predictions = svm_model_linear.predict(x_test)

    # model accuracy for X_test  
    accuracy = svm_model_linear.score(x_test, y_test)
    cr = classification_report(y_test, svm_predictions)
    # creating a confusion matrix
    print(cr)
    print(accuracy)

    # training a KNN classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 10).fit(x_train, y_train)
    # accuracy on X_test
    accuracy = knn.score(x_test, y_test)
    print (accuracy)


    
def decisionClassifiers():
    with open('image_decision_cv1_train.pkl', 'rb') as f: 
        x_train_image, Y_train = pickle.load(f)

    with open('image_decision_cv1_test.pkl','rb') as f:  
        x_test_image, Y_test = pickle.load(f)

    with open('image_decision_cv1_val.pkl','rb') as f: 
        x_val_image, Y_val = pickle.load(f)

    print("loaded images")
    with open('text_decision_cv1_train.pkl', 'rb') as f:  
        x_train_text = pickle.load(f)

    with open('text_decision_cv1_test.pkl','rb') as f:  
        x_test_text = pickle.load(f)

    with open('text_decision_cv1_val.pkl','rb') as f:  
        x_val_text = pickle.load(f)

    #arrange data
    x_train = []
    x_test = []
    x_val = []

    for i in range(0,len(x_test_image)):
        joint = np.concatenate([x_test_image[i],x_test_text[i]])
        x_test = np.concatenate([x_test,joint])

    for i in range(0,len(x_train_image)):
        joint = np.concatenate([x_train_image[i],x_train_text[i]])
        x_train = np.concatenate([x_train,joint])
         
    for i in range(0,len(x_val_image)):
        joint = np.concatenate([x_val_image[i],x_val_text[i]])
        x_val = np.concatenate([x_val,joint])

    x_val = x_val.reshape(len(x_val_image),12)
    x_test = x_test.reshape(len(x_test_image),12)
    x_train = x_train.reshape(len(x_train_image),12)

    # arrange y's for scikit learn format
    y_train = np.zeros(len(Y_train))
    y_test = np.zeros(len(Y_test))
    y_val = np.zeros(len(Y_val))

    for i in range (0,len(y_train)):
        y_train[i] = np.argmax(Y_train[i])
    for i in range (0,len(y_test)):
        y_test[i] = np.argmax(Y_test[i])  
    for i in range (0,len(y_val)):
        y_val[i] = np.argmax(Y_val[i])  



    # training an SVM classifier
    from sklearn.svm import SVC

    svm_model_linear = SVC(kernel = 'rbf', C = 1,max_iter=-1,probability=True,tol=1e-5,cache_size=400).fit(x_train, y_train)
    svm_predictions = svm_model_linear.predict(x_test)


    # model accuracy for X_test  
    accuracy = svm_model_linear.score(x_test, y_test)
    cr = classification_report(y_test, svm_predictions)
    cm = confusion_matrix(y_test,svm_predictions)
    print (cm)
    # creating a confusion matrix
    print(cr)
    print(accuracy)

    svm_model_linear = SVC(kernel = 'linear', C = 1,max_iter=-1,probability=True,tol=1e-5,cache_size=400).fit(x_train, y_train)
    svm_predictions = svm_model_linear.predict(x_test)


    # model accuracy for X_test  
    accuracy = svm_model_linear.score(x_test, y_test)
    cr = classification_report(y_test, svm_predictions)
    # creating a confusion matrix
    print(cr)
    print(accuracy)

    # training a KNN classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 10).fit(x_train, y_train)
    # accuracy on X_test
    accuracy = knn.score(x_test, y_test)
    print (accuracy)








