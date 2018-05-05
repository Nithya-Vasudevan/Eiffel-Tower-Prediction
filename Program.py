#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

#Step 1 = Convolution
classifier.add(Convolution2D(32,3,3, input_shape = (64, 64, 3), activation = 'relu'))

#Step 2 = Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step3 - Flattening
classifier.add(Flatten())

#Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#Mention Entire File Path
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

#Mention Entire File Path
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


classifier.fit_generator(training_set,
        steps_per_epoch=572,
        epochs=15,
        validation_data=test_set,
        validation_steps=572)

#3 Part 3 - Making new predictions


import numpy as np
from keras.preprocessing import image
imgName = input("Enter image name present in dataset folder(example:- image1):  ")


imgPath = "dataset/single_prediction/" + imgName + ".jpg"
print("Image Path: dataset/single_prediction/" + imgName + ".jpg")
test_image = image.load_img(imgPath, target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] ==1:
    prediction = 'Not an Eiffel Tower'
else:
    prediction = 'Eiffel Tower'
    
print("Result: " + prediction)