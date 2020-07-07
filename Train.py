
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix,classification_report
classes = 7   #this is the amount of facial expressions
rows = 48    #dimensions of images
col = 48    #dimensions of images
size = 32   #number of images to train at a time
training_samples = 28709 #number of samples available for training
test_samples = 3589 #number of samples available for validation
epochs = 500 #number of epochs
val = test_samples//size
##Prepare training and validation sets
data_training = '/home/chime/PycharmProjects/Testthesis/fer2013/images_fer2013/Training' #path of training data
data_testing = '/home/chime/PycharmProjects/Testthesis/fer2013/images_fer2013/PrivateTest'   #path of test data
##reshape and rescale the images to get more images from the images already present for better accuracy(Data augmentation)
data_training_generation = ImageDataGenerator(rescale=1./255, rotation_range=40,
                                              shear_range=0.2,zoom_range=0.2, width_shift_range=0.2,
                                              height_shift_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
data_testing_generation = ImageDataGenerator(rescale=1./255)
##generate the images to be used from the path and the reshaping done for  training and testing and send to the NN
data_training_generator = data_training_generation.flow_from_directory(data_training, color_mode='grayscale', target_size=(rows, col),
                                                                       batch_size=size, class_mode='categorical', shuffle=True)

data_testing_generator = data_testing_generation.flow_from_directory(data_testing, color_mode='grayscale',
                                                                     target_size=(rows, col), batch_size=size, class_mode='categorical', shuffle=True)

#CNN training
##Define model
model = Sequential()
##FIRST SET OF LAYERS
model.add(Conv2D(filters= 32,kernel_size= (3,3), padding='same', kernel_initializer='he_normal',
                 input_shape=(rows, col, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size= (3,3), padding='same', kernel_initializer='he_normal',
                 input_shape=(rows, col, 1), activation='relu'))
model.add(BatchNormalization())
##POOLING LAYER
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

##SECOND SET OF LAYERS
model.add(Conv2D(filters=64,kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
##POOLING LAYER
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

##THIRD SET OF LAYERS

model.add(Conv2D(filters= 128,kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128,kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation= 'relu'))
model.add(BatchNormalization())
##POOLING LAYER
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

## FOURTH SET OF LAYERS

model.add(Conv2D(filters=256,kernel_size=(3,3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256,kernel_size=(3,3), padding='same', kernel_initializer='he_normal',activation='relu'))
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
check = ModelCheckpoint('/home/chime/PycharmProjects/Testthesis/Emotions.h5',monitor='val_loss', mode='min', save_best_only=True, verbose=1)
## Stop training when performance measure stops improving after 5 times i.e patience
stop = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=400, verbose=1, restore_best_weights=True)

##monitor validation loss and reduce learning rate if validation loss does not improve for 5 epochs which means you are learning slowly
learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=400, verbose=1, min_delta=0.0001)
##Callback function to execute code
model_call_back = [check, stop, learning_rate]
##Compile the model
model.compile(loss='categorical_crossentropy', optimizer= Adam(learning_rate=0.001), metrics=['accuracy'])
#fit model
history = model.fit_generator(data_training_generator, steps_per_epoch=training_samples//size,
                              epochs= epochs, callbacks= model_call_back, validation_data= data_testing_generator,
                              validation_steps= val)
predictions= model.predict_classes(data_testing_generator)
confusion_matrix(data_training_generator,predictions)
print(classification_report(data_training_generator,predictions))