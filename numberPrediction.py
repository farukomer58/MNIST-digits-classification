import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import cv2  # Opencv import for reading images      
 
# Load data, dataset of keras with numbers
mnist = keras.datasets.mnist # 28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# Normalize / Scale the data so values 250 change to values between 0 and 1
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

# Show Data example first number
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

# Build Model 
model = keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu)) # 128 neurons, activation 'relu'
model.add(keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(keras.layers.Dense(10, activation=tf.nn.softmax)) # Output numbers 0-9

# Compile and FIT/TRAIN Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3) # epochs default is 1, how many times it runs

# Evaluate Scores on new Dataset - see if it is overfitting
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc) # loss 0.17651446163654327 acc 0.9753999710083008

# Save Trained model
# model.save('number_reader_model')
# newModel = keras.models.load_model('number_reader_model')

# Predict and save results
predictions = model.predict([x_test]) # We always need to pass in an array into predict
print(predictions[0])
realNumber = np.argmax(predictions[0])  # Get the real values from the list of predictions values for the first item / Transfer array into number
print(realNumber)                       # We see the first predictions is a 7

# Print result first item
plt.imshow(x_test[0], cmap = plt.cm.binary)
plt.show()

# Read image with opencv
# Predict Number from Image drawn in paint
img_path = '3.png'
img = cv2.imread(img_path, 0)       # read image as grayscale. Set second parameter to 1 if rgb is required 
img_reverted= cv2.bitwise_not(img)  
scaledImage = cv2.resize(img_reverted, (28,28)) 
new_img = scaledImage / 255.0  # Now all values are ranging from 0 to 1, where white equlas 0.0 and black equals 1.0 
new_img = new_img[None, :, : ] # Add Dimension at the start ndarray dimension 2d to 3d 

# Predict and show results
predictions = model.predict([new_img]) # We always need to pass in an array into predict
print(predictions[0])
realNumber = np.argmax(predictions[0]) # Get the real values from the list of predictions values for the first item / Transfer array into number
print(realNumber)                      # We see the first predictions 

# Print result first item
plt.imshow(new_img[0], cmap = plt.cm.binary)
plt.show()