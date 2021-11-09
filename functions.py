#imports
import cv2
import numpy as np
from time import time
import pandas as pd
import os

#!pip install imblearn

from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.metrics.pairwise import paired_distances
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
np.random.seed(seed=666)

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input,Conv2D, MaxPooling2D,BatchNormalization, LeakyReLU,concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_images(folder,names):
    """
    Creates the lists of images from the folder. Used in the load function.

    """
    images = []
    for id in names:
        img = cv2.imread(os.path.join(folder,str(id)+".bmp"))
        img_cyt = cv2.imread(os.path.join(folder,str(id)+"_segCyt.bmp"))
        img_nuc = cv2.imread(os.path.join(folder,str(id)+"_segNuc.bmp"))
        if img is not None:
            images.append([img,img_cyt,img_nuc])
    return images

def load(Working_directory) :
    """
    Load the different lists required from the folder Working_directory.
    Careful : The folder needs to be in the same format as the one in Kaggle (same structure and especially with the files metadataTrain.csv and SampleSubmission.csv.

    """
    data = pd.read_csv(Working_directory+"metadataTrain.csv")
    data_test = pd.read_csv(Working_directory+"SampleSubmission.csv")

    data_sorted_train = data.sort_values("ID")
    Y_train = data_sorted_train["ABNORMAL"]
    Id_train = data_sorted_train["ID"]
    Y_multi_train=data_sorted_train["GROUP"]
    Y_multi_train_cat = to_categorical(Y_multi_train)

    data_sorted_test = data_test.sort_values("ID")
    Id_test = data_sorted_test["ID"]
    Y_test = data_sorted_test["ABNORMAL"] ### Y_test will obviously be modified later on
    Y_multi_test=data_sorted_test["ABNORMAL"]


    Images_train = load_images(Working_directory+"/Train/Train/",Id_train)
    print("Shape Image Train : ",np.shape(np.array(Images_train,dtype=object)))
    Images_test = load_images(Working_directory+"/Test/Test/",Id_test)
    print("Shape Image Test : ",np.shape(np.array(Images_test,dtype=object)))
    return (Images_train,Y_train,Y_multi_train,Y_multi_train_cat,Images_test,Id_test)


def create_sol(Y,Id_test,problem="binary",title="Submission.csv") :
    """
    Creates the CSV document in a format ready for submission.

    """
    if problem == "binary" :
        name='ABNORMAL'
    elif problem == "multiclass" :
        name='GROUP'
    Id = Id_test
    data_frame = {'ID':np.array(Id),name:np.array(Y).astype(int)}
    data_frame = pd.DataFrame(data_frame)
    data_frame.to_csv(title,index=False)


def compute_features(Images) :
    """
    Computes 17 out of the 20 described features.

    """

    X=[]
    i=0
    for im in Images :
        i+=1
        features=[]
        cyt = im[1].copy()
        nuc = im[2].copy()
        cyt=cv2.cvtColor(cyt, cv2.COLOR_BGR2GRAY)
        nuc=cv2.cvtColor(nuc, cv2.COLOR_BGR2GRAY)


        ## 1st Feature :
        pixels_nuc = len(np.column_stack(np.where(im[2] > 0)))
        features.append(pixels_nuc)

        ## Feature n°2 :
        pixels_cyt = len(np.column_stack(np.where(im[1] > 0)))
        features.append(pixels_cyt)

        ## Feature n°3 :
        area_ratio = (pixels_nuc / (pixels_nuc+pixels_cyt)) * 100
        features.append(area_ratio)

        ## Features n°4-5 (Brightness)
        nuc_b = im[0].copy()
        nuc_b[np.where(im[2]==0)] = 0
        cyt_b = im[0].copy()
        cyt_b[np.where(im[1]==0)] = 0
        b = 0
        for k in nuc_b :
            b += np.sum(k) / (np.size(k))
        features.append(b)
        b = 0
        for k in cyt_b :
            b += np.sum(k) / (np.size(k))
        features.append(b)

        ## Features n°6-7-8-9
        # Find contours:

        contours_nuc, hierarchy = cv2.findContours(nuc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours_nuc==[] :## equivalent to no nuclear as some images apparently have no nuclear.
            features.append(0)
            features.append(0)
            features.append(0)
        else :
            if len(contours_nuc)>1 :
                l=[len(contours_nuc[j]) for j in range(len(contours_nuc))]
                k = np.argmax(l)
            else :
                k=0

            M = cv2.moments(contours_nuc[k])
            center=[round(M['m10'] / M['m00']),round(M['m01'] / M['m00'])]
            ## Features n°6-7 :
            d=[]
            for contour in contours_nuc[k] :
                d.append(paired_distances([center],contour))
            features.append(np.min(d))
            features.append(np.max(d))

            ##Elongation Not found

            ## Roundness :
            r = np.mean(d)
            dist=[]
            for di in d :
                dist.append(np.sqrt(r*r + di*di))
            features.append(np.mean(dist))


        cyt_t = cyt | nuc
        cyt_t = np.pad(cyt_t, pad_width=1, mode='constant',constant_values=0)

        # Find contours:
        contours_cyt, hierarchy = cv2.findContours(cyt_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Calculate image moments of the detected contour
        M = cv2.moments(contours_cyt[0])


        if round(M['m00']) == 0:
            cx = 0
            cy = 0
            for p in contours_cyt[0]:
                cx += p[0][0]
                cy += p[0][1]
            cx = round(cx/len(contours_cyt[0]))
            cy = round(cy/len(contours_cyt[0]))
            center=[cx,cy]
        else :
            center=[round(M['m10'] / M['m00']),round(M['m01'] / M['m00'])]
        ## Features n°10-11 :
        d=[]
        for contour in contours_cyt[0] :
            d.append(paired_distances([center],contour))
        features.append(np.min(d))
        features.append(np.max(d))

        ##Elongation : Not Found

        ## Roundness :
        r = np.mean(d)
        dist=[]
        for di in d :
            dist.append(np.sqrt(r*r + di*di))
        features.append(np.mean(dist))

        ## Features n°14-15 :
        features.append(len(contours_nuc))
        features.append(len(contours_cyt))

        ## Feature n°16 : Not made (the tries were full of strange bugs)

        ## Features n°17-18-19-20 :
        nuc = im[0].copy()
        nuc[np.where(im[2]==0)] = 0
        cyt = im[0].copy()
        cyt[np.where(im[1]==0)] = 0
        features.append(np.max(nuc))
        features.append(np.min(nuc))
        features.append(np.max(cyt))
        features.append(np.min(cyt))

        X.append(features)
    return(X)



def X_train_images(Images_train,Images_test,option=0) :
    """
    USED ONLY FOR THE DEEP LEARNING METHODS
    Create X_train and X_test in different forms depending on the value of option.
    If option = 0, we only consider the original images (not masks)
    If option = 1, we consider images + masks as one image (the masks are considered channels)
    If option = 2, the images and masks are separated (6 variables returned : 3 training and 3 testing)

    """
    X_train = np.array(Images_train)[:]
    X_test = np.array(Images_test)[:]


    X=np.zeros((X_train.shape[0],3,100,100))
    k=0

    for im in X_train :
        ima=[]
        for i in [0,1,2] :
            img = im[i]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image=cv2.resize(img_gray, (100,100), interpolation = cv2.INTER_AREA)
            ima.append(image)
        X[k] = ima
        k+=1

    X_train = X

    X=np.zeros((X_test.shape[0],3,100,100))
    k=0
    for im in X_test :
        ima=[]
        for i in [0,1,2] :
            img = im[i]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image=cv2.resize(img_gray, (100,100), interpolation = cv2.INTER_AREA)
            ima.append(image)
        X[k] = ima
        k+=1

    X_test = X


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255


    X_train = X_train.reshape((X_train.shape[0],X_train.shape[2],X_train.shape[3],3))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[2],X_test.shape[3],3))


    if option == 0 :
        return (X_train[:,:,:,0].reshape((X_train.shape[0],X_train.shape[1],X_train.shape[3],1)),
                X_test[:,:,:,0].reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1)))
    if option == 1 :
        return (X_train,X_test)
    if option == 2 :
        return (X_train[:,:,:,0].reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1)),
                X_train[:,:,:,1].reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1)),
                X_train[:,:,:,2].reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1)),
                X_test[:,:,:,0].reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1)),
                X_test[:,:,:,1].reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1)),
                X_test[:,:,:,2].reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1)))



def plot_pie(Y_multi) :
    """
    Plot the pie graph of the classes' proportions.

    """
    pie=[]
    for k in range(9):
        pie.append(np.count_nonzero(Y_multi == k))
    plt.figure(figsize=(7,7))
    explode = (0.1, 0.1, 0.2, 0.2,0.2,0.2,0.2,0.1,0.1)
    plt.pie(pie,labels=[0,1,2,3,4,5,6,7,8],autopct='%1.1f%%', shadow=True,explode=explode)

def preprocessing(X_train,Y_train,X_test,oversample=True) :
    """
    Scales the data and oversample it if required.

    """
    if oversample :
        ros = RandomOverSampler(random_state=0)
        X_train, Y_train = ros.fit_resample(X_train, Y_train)
    norm = MinMaxScaler()
    norm.fit(X_train)
    X_train = norm.transform(X_train)
    X_test = norm.transform(X_test)
    return(X_train,X_test,Y_train)

def feature_importance(model) :
    """
    Plot the graph of feature importances.

    """
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance)
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0])
    featfig = plt.figure()
    featax = featfig.add_subplot(1, 1, 1)
    featax.barh(sorted_idx, feature_importance[sorted_idx], align='center')
    featax.set_yticks(sorted_idx)
    featax.set_xlabel('Feature Importance (%)')
    plt.tight_layout()
    plt.show()

def models_opt(type,X_train,Y_train,X_test) :
    """
    Do some GridSearchCV on different models to give an optimal model for this problem. All the parameters given without GridSearchCV were computed with cross validation before, and are not used here to reduce the computation time.

    """
    X_train,X_test=np.array(X_train),np.array(X_test)
    if type == "knn" :
        knn = KNeighborsClassifier()
        parameters={'n_neighbors':[1,2,3,4,5,6,7,8,9],'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
        cv = GridSearchCV(knn,parameters)

    if type == "SVC" :
        svm = SVC(kernel='rbf')
        parameters = {'C':[0.01,0.1,1,10,100,1000,10000],'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1,'scale']}
        cv = GridSearchCV(svm,parameters)

    if type =="boosting" :
        parameters = {'min_samples_leaf':[1,2,3,4,5,6,7]}
        boost = GradientBoostingClassifier(n_estimators=500,learning_rate=0.15,max_depth=6,loss="deviance",         min_samples_split=9,max_features='log2',criterion='friedman_mse')
        cv=GridSearchCV(boost,parameters)

    if type =="random_forest" :
        forest = RandomForestClassifier(max_features='auto',criterion='gini',max_depth=8)
        parameters = {'n_estimators':np.logspace(0, 3, num=4,dtype=int)}
        cv = GridSearchCV(forest,parameters)
    cv.fit(X_train,Y_train)
    Y_test=cv.predict(X_test)
    return(Y_test,cv)

def opt_boost_model(X_train,Y_train) :
    """
    Gives the optimal boosting classifier for this problem (all the parameters's been found by GridSearchCV).

    """
    model = GradientBoostingClassifier(n_estimators=500,learning_rate=0.15,max_depth=6, loss="deviance",
                                            min_samples_split=9,min_samples_leaf=6,
                                            max_features='log2',criterion='friedman_mse')
    model.fit(X_train,Y_train)
    return (model)


def gridcv_boosting(X_train,Y_train) :
    """
    Gives the best possible boosting algorithm with these numbers of estimators (increasing n_estimator will bring a slightly better score). All the parameters have been found with gridsearchCV. The reason I am using GridSearchCV for only two paramaters is that it takes a very long time to do cross validations on all the parameters.
    Also plots the evolution of the scores during gridsearch.

    """
    n_estimators=[1,5,10,20,30,40,50,60,70,100]
    max_features=['log2','sqrt',None]
    parameters = {'n_estimators':[1,5,10,20,30,40,50,60,70,100],'max_features':['log2','sqrt',None]}
    cv=GridSearchCV(estimator=GradientBoostingClassifier(min_samples_leaf=6,
                        learning_rate=0.15,max_depth=6,loss="deviance", min_samples_split=9,
                        criterion='friedman_mse'),
                        param_grid = parameters, scoring='accuracy',n_jobs=4, cv=5)
    cv.fit(X_train,Y_train)
    scores_mean = cv.cv_results_['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(max_features),len(n_estimators))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    for idx, val in enumerate(max_features):
        ax.plot(n_estimators, scores_mean[idx,:], '-o', label= "Max Feature" + ': ' + str(val))

    ax.set_title("Grid Search Scores")
    ax.set_xlabel('N Estimators')
    ax.set_ylabel('CV Average Score')
    ax.legend(loc="best")
    ax.grid('on')


def first_cnn() :

    model = Sequential()

    model.add(Conv2D(32, (4, 4), input_shape = (100,100,1), activation = 'relu',padding="same"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(32, (4, 4), activation = 'relu',padding="same"))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    model.compile(optimizer = optimizers.Adam(lr=0.01), loss = 'binary_crossentropy', metrics =['accuracy'])
    model.summary()
    return(model)

def ann(X_train) :

    n_input = X_train.shape[1]

    model_mlp_multi_layer = Sequential()
    model_mlp_multi_layer.add(Dense(128,activation='relu',input_shape=(n_input,)))
    model_mlp_multi_layer.add(BatchNormalization())
    model_mlp_multi_layer.add(Dropout(0.5))
    model_mlp_multi_layer.add(Dense(128,activation='relu'))
    model_mlp_multi_layer.add(BatchNormalization())
    model_mlp_multi_layer.add(Dense(128,activation='relu'))
    model_mlp_multi_layer.add(BatchNormalization())
    model_mlp_multi_layer.add(Dropout(0.5))
    model_mlp_multi_layer.add(Dense(1,activation='sigmoid'))

    learning_rate = 0.001
    model_mlp_multi_layer.compile(loss='binary_crossentropy',  optimizer=optimizers.Adam(lr=0.01),metrics=["accuracy"])
    model_mlp_multi_layer.summary()
    return(model_mlp_multi_layer)



def create_convolution_layers(input_img,input_shape):
    """
    Creates the convolution part in the final model (see the image in the report).
    Made to simplify the function final_model.

    """
    model = Conv2D(16, (3, 3), padding='same', input_shape=input_shape)(input_img)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPooling2D((2, 2),padding='same')(model)
    model = Dropout(0.25)(model)

    model = Conv2D(32, (3, 3), padding='same', input_shape=input_shape)(input_img)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPooling2D((2, 2),padding='same')(model)
    model = Dropout(0.25)(model)

    model = Conv2D(64, (3, 3), padding='same')(model)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
    model = Dropout(0.25)(model)

    model = Conv2D(128, (3, 3), padding='same')(model)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
    model = Dropout(0.4)(model)

    model = Conv2D(256, (3, 3), padding='same')(model)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
    model = Dropout(0.4)(model)
    model = Flatten()(model)

    return model

def final_model(X_train_0,X_train_1,X_train_2,X_train_3,Y_train) :
    """
    Creates AND fit the final model (for 2 epochs ; increase the number of epochs to better the results
    X_train_0 is the array of original images
    X_train_1 is the array of cytoplasm masks
    X_train_2 is the array of nucleus masks
    X_train_3 is the array of the computed features

    """
    shape0 = np.array(X_train_0)[0].shape
    img0 = Input(shape=shape0)
    img0_model = create_convolution_layers(img0,shape0)

    shape1 = np.array(X_train_1)[0].shape
    img1 = Input(shape=shape1)
    img1_model = create_convolution_layers(img1,shape1)

    shape2 = np.array(X_train_2)[0].shape
    img2 = Input(shape=shape2)
    img2_model = create_convolution_layers(img2,shape2)

    shape3 = np.array(X_train_3)[0].shape
    features = Input(shape=shape3)
    features_model = Dense(512)(features)
    dense = BatchNormalization()(features)
    dense= LeakyReLU(alpha=0.1)(features)
    dense = Dropout(0.5)(features)

    features_model = Flatten()(features_model)

    conv = concatenate([img0_model,img1_model,img2_model])


    dense = Dense(1024)(conv)
    dense = LeakyReLU(alpha=0.1)(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[img0,img1,img2,features], outputs=[output])
    opt = optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    model.summary()
    model.fit([X_train_0,X_train_1,X_train_2,np.array(X_train_3)],Y_train,epochs=2,batch_size=32)
    return(model)


def convert_Y_test(Y_test,problem="binary") :
    if problem =="binary" :
        Y_test=Y_test.reshape((Y_test.shape[0]))
        for k in range(len(Y_test)):
            Y_test[k]=round(Y_test[k])
    if problem == "multiclass" :
        Y_test = np.argmax(Y_test,axis=1)
    return(Y_test)

