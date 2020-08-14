
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix,classification_report
classes = 7   #this is the amount of facial expressions
rows = 48    #dimensions of images
col = 48    #dimensions of images
size = 32 #number of images to train at a time
training_samples = 28709 #number of samples available for training
test_samples = 3589 #number of samples available for validation
epochs = 1 #number of epochs
val = test_samples//size #validation steps
stepsepochs=training_samples//size #number of steps per epoch
##Prepare training and validation sets
data_training = '/home/chime/PycharmProjects/Testthesis/fer2013/images_fer2013/Training' #path of training data
data_testing = '/home/chime/PycharmProjects/Testthesis/fer2013/images_fer2013/PrivateTest'   #path of test data
##reshape and rescale the images to get more images from the images already present for better accuracy(Data augmentation)

data_training_generation = ImageDataGenerator(rescale=1./255, rotation_range=40,
                                              shear_range=0.2,zoom_range=0.2, zca_whitening=False,width_shift_range=0.3,
                                              height_shift_range=0.3,data_format=None,validation_split=0.0,
                                              horizontal_flip=True, vertical_flip=True, brightness_range=None,
                                              fill_mode='nearest', preprocessing_function=None)

data_testing_generation = ImageDataGenerator(rescale=1./255)
##generate the images to be used from the path and the reshaping done for  training and testing and send to the NN
##color_mode is grayscale because the images are in grayscale
##batchsize is the numer of images to be used from the generator per batch and this can be changed
##class mode is categorical because we have about 7 classes
##shuffle is true because we intend on shuffling the order of the yielded images
##seed is for applying random image augmentation
data_training_generator = data_training_generation.flow_from_directory(data_training, target_size=(rows, col),color_mode='grayscale',
                                                                        batch_size=size,class_mode='categorical', shuffle=True,seed=42,
                                                                         follow_links=False,subset=None)

data_testing_generator = data_testing_generation.flow_from_directory(data_testing, target_size=(rows, col),color_mode='grayscale',
                                                                      batch_size=size, class_mode='categorical', shuffle=True,seed=42,
                                                                     follow_links=False, subset=None)

#CNN training
##Define model
model = Sequential()
##FIRST SET OF LAYERS
## The code for the convolutional layer is a universal code for using relu as an activation function
## Most develops add one after the other for conv2d but i prefer adding everything to the model at once to make it more readable
## strides can be ignored as default is (1,1) but i'd rather use it to ensure the code is running at optimum
model.add(Conv2D(filters= 32,kernel_size= (3,3), strides=(1,1),padding='same', kernel_initializer='he_normal',
                 input_shape=(rows, col, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size= (3,3), strides=(1,1),padding='same', kernel_initializer='he_normal',
                 input_shape=(rows, col, 1), activation='relu'))
model.add(BatchNormalization())
##POOLING LAYER
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

##SECOND SET OF LAYERS
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=(3,3), strides=(1,1),padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
##POOLING LAYER
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

##THIRD SET OF LAYERS

model.add(Conv2D(filters= 128,kernel_size=(3,3), strides=(1,1),padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1), padding='same', kernel_initializer='he_normal', activation= 'relu'))
model.add(BatchNormalization())
##POOLING LAYER
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

## FOURTH SET OF LAYERS

model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1), padding='same', kernel_initializer='he_normal',activation='relu'))
model.add(BatchNormalization())
##POOLING LAYER
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
## FIFTH SET OF LAYERS

model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=512,kernel_size=(3,3), strides=(1,1),padding='same', kernel_initializer='he_normal',activation='relu'))
model.add(BatchNormalization())
##POOLING LAYER
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#FLATTEN  IMAGES
model.add(Flatten())
model.add(Dense(64, kernel_initializer= 'he_normal', activation= 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#SIXTH LAYER
model.add(Dense(64, kernel_initializer='he_normal', activation= 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#SEVENTH LAYER(CLASSIFIER LAYER)
model.add(Dense(classes, kernel_initializer='he_normal', activation= 'softmax'))


print(model.summary())


##Callback functions
##modelcheckpoint to save the model with the best performance
##For this model we are monitoring the validation accuracy and saving the maximum value of the validation accuracy
check = ModelCheckpoint(filepath='/home/chime/PycharmProjects/Testthesis/Emotions.h5',monitor='val_accuracy', verbose=1 ,save_best_only=True, mode='max')
## Stop training when performance measure stops improving after patience times i.e patience
##The measure being monitored is the validation loss
##min_delta is the minimum chnage thet can be considered an improvement
stop = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=2400, verbose=1, baseline=None,restore_best_weights=True)


##monitor validation loss and reduce learning rate if validation loss does not improve for patience  which means you are learning slowly

learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2400, verbose=1, min_delta=0.0001)


##Callback function to execute code
model_call_back = [check, stop, learning_rate]
##Compile the model
model.compile(loss='categorical_crossentropy', optimizer= Adam(learning_rate=0.001), metrics=['accuracy'])
#fit model
history = model.fit_generator(data_training_generator, steps_per_epoch=stepsepochs,
                              epochs= epochs, callbacks= model_call_back, validation_data= data_testing_generator,
                              validation_steps= val)
#predictions= model.predict_classes(data_testing_generator,verbose=1)
#confusion_matrix(data_training_generator,predictions)
#print(classification_report(data_training_generator,predictions))
#model.save_weights('/home/chime/PycharmProjects/Testthesis/Emotions.h5')