from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from class_names import cifar_100_coarse_classes
from tensorflow.keras.utils import to_categorical


# n_classes must be <= 20. In 1909.11572 they use only 3 coarse classes
n_classes = 3
learning_rate = 0.0129
batch_size = 128
# Define model for training on the coarse classes
# Defining total DOF and penultimate layer size:
n_DOF = 1024
n_penul = 1024
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
(train_images_coarse, train_labels_coarse), (test_images_coarse, test_labels_coarse) = datasets.cifar100.load_data(
    label_mode='coarse')

# Filter based on class labels


train_filter = np.where(train_labels_coarse < 3)[0]
test_filter = np.where(test_labels_coarse < 3)[0]

train_images, train_labels = train_images[train_filter], train_labels[train_filter]
test_images, test_labels = test_images[test_filter], test_labels[test_filter]

train_images_coarse, train_labels_coarse = train_images_coarse[train_filter], train_labels_coarse[train_filter]
test_images_coarse, test_labels_coarse = test_images_coarse[test_filter], test_labels_coarse[test_filter]

class_names = [coarse_label for coarse_label in cifar_100_coarse_classes[:n_classes]]

fine_classes = list(set(test_labels.reshape(-1).tolist()))
n_fine_class = len(fine_classes)

train_labels = np.array([fine_classes.index(label) for label in train_labels])
test_labels = np.array([fine_classes.index(label) for label in test_labels])

# Sanity Check: plot examples of each coarse class
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    assert np.all(train_images[i] == train_images_coarse[i])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels_coarse[i][0]])
plt.show()

# Normalize pixel values to be between 0 and 1
train_images_coarse, test_images_coarse = train_images_coarse / 255.0, test_images_coarse / 255.0
train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential()
model.add(layers.Conv2D(256, (3, 3), activation=leaky_relu, input_shape=train_images[0].shape))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation=leaky_relu))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(256, (3, 3), activation=leaky_relu))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation=leaky_relu))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
# Dense Layers
model.add(layers.Dense(1024, activation=leaky_relu))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_penul, activation=leaky_relu))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_classes, activation='softmax'))

model.summary()

sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=.9)
adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    # optimizer=sgd,
    optimizer=adam,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(train_images_coarse, train_labels_coarse, epochs=100,
                    validation_data=(test_images_coarse, test_labels_coarse),
                    batch_size=batch_size,
                    shuffle=True)

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
model_2.add(layers.Dense(n_fine_class, activation='softmax'))

print("summary after adding new layer")
model_2.build(input_shape=model.input_shape)
model_2.summary()
model_2.compile(optimizer=adam,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

history = model_2.fit(train_images, train_labels, epochs=100
                      ,
                      shuffle=True,
                      validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model_2.evaluate(test_images, test_labels, verbose=2)

print(test_acc)
plt.show()
