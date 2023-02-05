#Import modules and libraries
import os
import sys
import glob
import time
import datetime



import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

from keras.metrics import AUC
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

def load_data(image_paths, split, label_path=None, scale='none'):
    '''
    Read a set from a path.
    
    This function returns a batch of images and sometimes labels from a list of filepaths 
    containing said images. The function also scales the dataset by a specified factor.

    Parameters
    ----------
    image_paths : list of str
        List of directories containing the images.
    split : str
        Type of set to be loaded. One of either 'train', 'val,' or 'test.'
    label_path : str, default None
        Path to csv file containing image labels. Set to None if split == 'test.'
    scale: str, default 'none'
        Factor to scale the dataset with. Should be one of 'none,' 'minmax,' 'standard,'
        'normalized,' or 'var_coeff.'

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray) or numpy.ndarray
        Arrays of images of loaded images (and labels) loaded.
    '''
    #For training images
    if split == 'train':

        #Create an empty list for images
        images = []
        #Read labels from csv file path as a numpy array
        labels = pd.read_csv(label_path, index_col='id').to_numpy()
        
        #Read image all images in path as grayscale and append to list
        for path in image_paths:
            image = cv2.imread(path, 0)
            images.append(image)

        #If a directory to store augmented images exists in folder
        if os.path.isdir(os.path.join(os.getcwd(),'Aug_Images')):
            for path in glob.glob((os.path.join(os.getcwd(),'Aug_Images', '*'))): 
                os.remove(path) #remove all images in directory

        else:
            #Create a directory for augmented images
            os.mkdir('Aug_Images') 

        #Instantiate an ImageDataGenerator object for data augmentation   
        augmentor = ImageDataGenerator(horizontal_flip=True, vertical_flip=False, rotation_range=15, width_shift_range=0.2, 
            height_shift_range=0.2, brightness_range=[0.1, 1.0], shear_range=0.2)
        X = np.array(images) # convert the images temporarily to an array
        
        #Generate augmented images and labels and save images to Aug_Images
        for X_batch, y_batch in augmentor.flow(X.reshape(X.shape[0],64,64,1), labels, batch_size=3000, save_to_dir='Aug_Images', save_format='png'):
            break 
        labels = np.append(labels, y_batch, axis=0) #append augmented labels to the original labels

        #Read all images from Aug_Images and add to the original image list
        aug_path = glob.glob((os.path.join(os.getcwd(),'Aug_Images', '*.png')))
        for path in aug_path:
            image = cv2.imread(path, 0)
            images.append(image)

        #Convert image_list to an array
        features = np.array(images)
        assert len(features) == len(labels), f'Number of images and labels do not match. Found {len(features)},images and {len(labels)}, labels.'
        print(f'Found {len(image_paths)} training images and {len(aug_path)} augmented images')

    #For validation images
    elif split == 'val':

        #Create an empty list for images
        images = []
        #Read labels from csv file path as a numpy array
        labels = pd.read_csv(label_path, index_col='id').to_numpy()

        #Read image all images in path as grayscale and append to list
        for path in image_paths:
            image = cv2.imread(path, 0)
            images.append(image)
        
        #Convert image_list to an array
        features = np.array(images)
        assert len(features) == len(labels), f'Number of images and labels do not match. Found {len(features)},images and {len(labels)}, labels.'
        print(f'Found {len(image_paths)} validation images.')

    #For test images
    elif split == 'test':

        #Create an empty list for images
        images = []

        #Read image all images in path as grayscale and append to list
        for path in image_paths:
            image = cv2.imread(path, 0)
            images.append(image)
        
        #Convert image_list to an array
        features = np.array(images)
        print(f'Found {len(image_paths)} test images.')
    
    #For an invalid imput for split
    else:
        raise ValueError("Value for split has to be either of \'train\', 'val\', or \'test\'.")

    if scale == 'none': 
        #Use unscaled images as features
        features_norm = features

    elif scale == 'minmax':
        #Use a minmax-type scaler on the features
        features_norm = (features)/255.

    elif scale == 'normalized':
        #Scale all features to have values between -0.5 and 0.5
        features_norm = (features)/255. - 0.5

    elif scale == 'standard':
        features_norm = (features - features.mean())/features.std()

    elif scale == 'var_coeff':
        #Use a standard-type scaler on the features
        features_norm = (features)*features.mean()/features.std()

    #For an invalid imput for scale
    else:
        raise ValueError('Invalid scale value used. Values have to be one of either \'none\', \'minmax\',\'normalized\', or \'var_coeff\'.')

    #For train and val splits return features with labels 
    if split != 'test':
        return features_norm, labels

    #For test split return only features
    else:
        return features_norm

def build_model():
    '''
    Create a model as a list of layers.
    
    This function returns sequential VGG-type model as a list of stacked layers.

    Returns
    -------
    keras.engine.sequential.Sequential
        A sequential model.
    '''
    initializer = 'he_uniform'
    model = Sequential()
    
    model.add(Conv2D(32,kernel_size=5, padding='same', activation='relu', input_shape=(64, 64, 1), kernel_initializer=initializer))
    model.add(MaxPooling2D((4,4), strides=2, padding='same'))
    model.add(Dropout(rate=0.25))
    
    model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu', kernel_initializer=initializer))
    model.add(MaxPooling2D((4,4), strides=2, padding='same'))
    model.add(Dropout(rate=0.25))
        
    model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu', kernel_initializer=initializer))   
    model.add(MaxPooling2D((4,4), strides=2, padding='same'))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(256, kernel_size=5, padding='same', activation='relu', kernel_initializer=initializer))    
    model.add(MaxPooling2D((4,4), strides=2, padding='same'))   
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(512, kernel_size=5, padding='same', activation='relu', kernel_initializer=initializer))    
    model.add(MaxPooling2D((4,4), strides=2, padding='same'))   
    model.add(Dropout(rate=0.25))

    model.add(Flatten())    
    model.add(Dense(1024, activation='relu'))    
    model.add(Dropout(rate=0.5))    
    model.add(Dense(3, activation='sigmoid'))

    return model

def build_model_2():
    '''
    Create a model as a list of layers.
    
    This function returns a sequential VGG-type model as a list of stacked layers.

    Returns
    -------
    keras.engine.sequential.Sequential
        A sequential model.
    '''
    initializer = 'he_uniform'
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5,1), activation='relu', padding='same', input_shape=(64, 64, 1), kernel_initializer=initializer))
    model.add(Conv2D(32, kernel_size=(1,5), activation='relu', padding='same', kernel_initializer=initializer))
    model.add(MaxPooling2D((3,3), strides=2, padding='same'))

    model.add(Conv2D(64, kernel_size=(5,1), activation='relu', padding='same', kernel_initializer=initializer))
    model.add(Conv2D(64, kernel_size=(1,5), activation='relu', padding='same', kernel_initializer=initializer))
    model.add(MaxPooling2D((3,3), strides=2, padding='same')) 
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(128, kernel_size=(5,1), activation='relu', padding='same', kernel_initializer=initializer))
    model.add(Conv2D(128, kernel_size=(1,5), activation='relu', padding='same', kernel_initializer=initializer))
    model.add(MaxPooling2D((3,3), strides=2, padding='same')) 
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(256, kernel_size=(5,1), activation='relu', padding='same', kernel_initializer=initializer))
    model.add(Conv2D(256, kernel_size=(1,5), activation='relu', padding='same', kernel_initializer=initializer))
    model.add(MaxPooling2D((3,3), strides=2, padding='same')) 
    model.add(Dropout(rate=0.25))

    model.add(Flatten())    
    model.add(Dense(512, activation='relu'))   
    model.add(Dropout(rate=0.5))    
    model.add(Dense(3, activation='sigmoid'))

    return model

def test_preds(model, data, threshold):
    '''
    Make predictions from a model.
    
    This function returns an array of binary ints from as a model's prediction on a dataset.
    All float values below the the threshold are rounded down to 0, and those above,
    rounded up to 1.

    Parameters
    ----------
    model : keras.engine.sequential.Sequential
        Keras sequential model.
    data : numpy.ndarray
        An array of features.
    threshold : float
        A value to set degree of confidence on predictions.
    
    Returns
    -------
    numpy.ndarray
        An array of predictions.
    '''
    preds = model.predict(data)
    #Set all values > threshold to 1, else 0
    preds = np.where(preds > threshold, 1, 0) 

    return preds

def make_submission(model, data, name, threshold):    
    '''
    Create anad save a csv file for predictions.
    
    This function makes predictions, creates a Pandsa dataframe of the predictions,
    and save them to direcoty as a csv file. All predicted values above the threshold
    will be rounded up to 1, and those below will be rounded down to 0.

    Parameters
    ----------
    name : str
        A name for the csv file.
    threshold : float
        A value for the degree of confidence. 
    
    Returns
    -------
    None
    '''

    #Get a list of all images in the test file directory
    filespath = os.path.join(os.getcwd(), 'dl-2022-medical', 'test_images')
    #Create an empty list for the list indexes
    id = []

    #Add the filenames of all png files in the directory as indexes
    for file in os.listdir(filespath):
            if (file.endswith(".png")):
                id.append(file)
    
    #Create a Pandas series of the indexes
    ids = pd.Series(id, name = 'id')
    #Mak3 and convert model predictions to binary
    preds = test_preds(model, data, threshold)
    #Create a Pandas dataframe of the predictions
    df = pd.DataFrame(preds, columns=['label1', 'label2', 'label3'], index=ids)
    #Save predictions as csv file
    df.to_csv(os.path.join(os.getcwd(), 'Submissions', name), index=True)

def make_report(model, features, labels, threshold=0.5):
    '''
    Create classification report for each class per label.
    
    The function prints put a classification report for predicted feature values.
    It has threshold values to adjust values.

    Parameters
    ----------
    model : keras.engine.sequential.Sequential
        A sequential model.
    features : numpy.ndarray
        An array of features.
    labels : numpy.ndarray
        An array of features.
    threshold : float, default 0.5
        A value for the degree of confidence. 
    
    Returns
    -------
    None
    '''
    pred = model.predict(features)
    true = labels

    true_1, true_2, true_3 = np.hsplit(true, 3)
    pred_1, pred_2, pred_3 = np.hsplit(pred, 3)

    pred_1 = np.where(pred_1 > threshold, 1, 0 )
    pred_2 = np.where(pred_2 > threshold, 1, 0 )
    pred_3 = np.where(pred_3 > threshold , 1, 0 )

    print(classification_report(true_1, pred_1))
    print(classification_report(true_2, pred_2))
    print(classification_report(true_3, pred_3))

def info(file, text):
    '''
    Create and save a txt file from a multiline string.
    
    The function takes in a multiline string and creates a test file from it.
    The string contains vital information about the model, especially its 
    architecture and parameters.

    Parameters
    ----------
    file : str
        A name to save as txt file.
    text : str
        A multiline string containing model info. 
    
    Returns
    -------
    None
    '''
    
    #Create an empty txt file
    with open(os.path.join(os.getcwd(), 'Info', file +'.txt'), 'w') as f:
        for line in text: #add every line from the string to the txt file
            f.write(line)
            
def summarize_diagnostics(history):
    '''
    Make a graph of losses against AUC.
    
    Parameters
    ----------
    history : keras.callbacks.History
        A history object containing model info

    Returns
    -------
    None
    '''
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.subplots_adjust(bottom=0.7, top=2)
	
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['auc'], color='blue', label='train')
    plt.plot(history.history['val_auc'], color='orange', label='test')
    
    plt.legend()
	# save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')

    plt.show()
    
def main():
    start = time.time()
    #LOADING IMAGES
    begin = time.time()
    print('\n')
    print('------------------------Loading Images----------------------')

    #Get image paths as list of filepaths
    train_image_paths = glob.glob(os.path.join(os.getcwd(), 'dl-2022-medical', 'train_images', '*.png'))
    val_image_paths = glob.glob(os.path.join(os.getcwd(), 'dl-2022-medical', 'val_images', '*.png'))
    test_image_paths = glob.glob(os.path.join(os.getcwd(), 'dl-2022-medical', 'test_images', '*.png'))

    #Get labels paths
    train_label_paths = os.path.join(os.getcwd(), 'dl-2022-medical', 'train_labels.csv')
    val_label_paths = os.path.join(os.getcwd(), 'dl-2022-medical', 'val_labels.csv')

    #Loading images and labels using helper functions created.
    X_train, y_train = load_data(train_image_paths, 'train',train_label_paths, scale='minmax')
    X_val, y_val = load_data(val_image_paths,'val', val_label_paths, scale='minmax')
    X_test = load_data(test_image_paths, 'test', scale='minmax')

    loading_time = time.time() - begin
    print('Loaded data in', str(datetime.timedelta(seconds=loading_time)))

    print('---------------------Completed Loading Images-------------------')
    
    #Build model 1
    print('------------------------ Building Model 1 -----------------------')
    begin = time.time()
    model = build_model()
    auc = AUC(curve='PR', multi_label=True, num_labels=3)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
    lr = ReduceLROnPlateau(patience=2)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=256, epochs=10, callbacks=[lr, es])

    summarize_diagnostics(history)
    
    model_1_time = time.time() - begin
    print('Built model 1 in', str(datetime.timedelta(seconds=model_1_time)))

    print('--------------------- Completed Building Model 1 -------------------')
    #MAKE SUBMISSION FILe
    print('------------------------Making Submission File----------------------')
    make_submission(model, X_test, '05-01-02.csv', 0.6)

    model_info = '''
        train_size = 9035
        Threshold = 0.5
        scale = minmax
        aug_size = 3000
        augmentor = ImageDataGenerator(horizontal_flip=True, vertical_flip=False, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, brightness_range=[0.1, 1.0],
                shear_range=0.2), seed=42

        def build_model():
            model = Sequential()
            
            model.add(Conv2D(32,kernel_size=5, padding='same', activation='relu', input_shape=(64, 64, 1)))
            model.add(MaxPooling2D((3,3), strides=2, padding='same')) 
            model.add(Dropout(rate=0.2))

            model.add(Conv2D(64, kernel_size=5, activation='relu'))
            model.add(MaxPooling2D((3,3), strides=2, padding='same')) 
            model.add(Dropout(rate=0.2))
            
            model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))   
            model.add(MaxPooling2D((3,3), strides=2, padding='same')) 
            model.add(Dropout(rate=0.2))

            model.add(Conv2D(256, kernel_size=5, activation='relu'))    
            model.add(MaxPooling2D((3,3), strides=2, padding='same')) 
            model.add(Dropout(rate=0.2))

            model.add(Flatten())    
            model.add(Dense(512, activation='relu'))    
            model.add(Dropout(rate=0.5))    
            model.add(Dense(3, activation='sigmoid'))

            return model
        '''
    info('02-01-02', model_info)

    print('------------------------Classification Report For Model 1----------------------')

    make_report(model, X_val, y_val, 0.5)
    
    #--------------------------------------------------------------------------------------#
    
    print('\n')
    #Make model 2
    print('-------------------------------- Building Model 2 ------------------------------')

    begin = time.time()

    model = build_model_2()
    auc = AUC(curve='PR', multi_label=True, num_labels=3)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
    lr = ReduceLROnPlateau(patience=2)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=128, epochs=5, callbacks=[lr, es])

    model_2_time = time.time() - begin
    print('Built model 2 in', str(datetime.timedelta(seconds=model_2_time))) 

    print('--------------------------- Completed Building Model 2 -------------------------')
    
    #Make submission file
    print('------------------------Making Submission File----------------------')
    make_submission(model, X_test, '05-01-02.csv', 0.6)
    
    #Make classification report
    print('------------------------Classification Report For Model 2----------------------')
    make_report(model, X_val, y_val, 0.5)
    

    print('The script ran in', str(datetime.timedelta(seconds=time.time()-start)))

if __name__ == '__main__':
    main()