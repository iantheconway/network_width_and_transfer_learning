from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from class_names import cifar_100_classes
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
(train_images_coarse, train_labels_coarse), (test_images_coarse, test_labels_coarse) = datasets.cifar100.load_data(
    label_mode='coarse')

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images_coarse / 255.0, test_images_coarse / 255.0

class_names = [coarse_label for coarse_label in cifar_100_classes]

# Sanity Check: plot examples of each coarse class
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels_coarse[i][0]])
plt.show()

# Define model for training on the coarse classes
# Defining total DOF and penultimate layer size:
n_DOF = 1024
n_penul = 512
# Final layer size determined by number of classes:
n_classes = 20



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(np.floor_divide(n_DOF, n_penul), activation='relu'))
model.add(layers.Dense(n_penul, activation='relu'))
model.add(layers.Dense(n_classes, activation='softmax'))

model.summary()

#ADAM = tf.keras.optimizers.Adam(learning_rate=0.0129, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images_coarse, train_labels_coarse, epochs=10,
                    validation_data=(test_images_coarse, test_labels_coarse))

print("training on fine classes")

# freeze layers from coarse class model

for layer in model.layers:
    layer.trainable = False

print("summary before removing layer")
model.summary()

# Create a new model for training on the fine classes
model_2 = models.Sequential()

for layer in model.layers[:-1]:
    model_2.add(layer)

print("summary before adding new layer")

# Add a trainable final layer
model_2.add(layers.Dense(100, activation='softmax'))

print("summary after adding new layer")
model_2.build(input_shape=model.input_shape)
model_2.summary()
model_2.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = model_2.fit(train_images, to_categorical(train_labels), epochs=10,
                      validation_data=(test_images, to_categorical(test_labels)))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(test_images_coarse, test_labels_coarse, verbose=2)

print(test_acc)
plt.show()
