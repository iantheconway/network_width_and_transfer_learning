from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import datasets, layers, models
import wandb
from wandb.keras import WandbCallback

from class_names import cifar_100_coarse_classes


parser = argparse.ArgumentParser(description='Compare network architecture for transfer learning')
parser.add_argument('-n', '--penultimate_layer_dim',
                    metavar='n', type=int, nargs='+', default=[1024],
                    help='number of nodes for final hidden layer')

parser.add_argument('-e', '--epochs', nargs='?', const=100, type=int, default=100,
                    help='number of epochs to train for'
                    )

parser.add_argument('-t', '--use_transfer_learning', action='store_true',
                    help='weather to randomly initialize or use transfer learning')

parser.add_argument('-b', metavar='b', type=bool, nargs='+', default=False,
                    help='weather to use batch normalization')

parser.add_argument('-d', metavar='d', type=bool, nargs='+', default=False,
                    help='weather to use dropout')


def run_transfer_learning(transfer=True,
                          penultimate_layer_dim=1024,
                          batch_norm=False,
                          batch_size=128,
                          learning_rate=0.0129,
                          learning_rate_fine=0.001,
                          epochs=100,
                          dropout=False):
    """trains a model on a subset of the cifar 100 classes,
    freezes parameters and then does transfer learning on the remaining classes"""

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    config = {'transfer': transfer,
              'penultimate_layer_dim': penultimate_layer_dim,
              'batch_norm': batch_norm,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'learning_rate_fine': learning_rate_fine,
              'dropout': dropout}

    wandb.init(project="network_width_and_transfer_learning",
               sync_tensorboard=True,
               entity="iantheconway",
               config=config
               )

    # n_classes must be <= 20. In 1909.11572 they use only 3 coarse classes
    n_classes = 3

    relu = tf.keras.activations.relu

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
    (train_images_coarse, train_labels_coarse), (test_images_coarse, test_labels_coarse) = datasets.cifar100.load_data(
        label_mode='coarse')

    datagen.fit(train_images_coarse)

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
    wandb.log({"coarse_class_examples": plt})

    # Normalize pixel values to be between 0 and 1
    train_images_coarse, test_images_coarse = train_images_coarse / 255.0, test_images_coarse / 255.0
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation=relu, input_shape=train_images[0].shape, padding='same'))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation=relu, padding='same'))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    if dropout:
        model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation=relu, padding='same'))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation=relu, padding='same'))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    if dropout:
        model.add(layers.Dropout(0.25))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    # Dense Layers
    # model.add(layers.Dense(1024, activation=leaky_relu))
    if batch_norm:
        model.add(layers.BatchNormalization())
    if dropout:
        model.add(layers.Dropout(0.25))
    model.add(layers.Dense(penultimate_layer_dim, activation=relu))
    if batch_norm:
        model.add(layers.BatchNormalization())
    if dropout:
        model.add(layers.Dropout(0.25))
    model.add(layers.Dense(n_classes, activation='softmax'))

    model.summary()

    sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=.9)
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=sgd,
        # optimizer=adam,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    if transfer:
        model.fit(datagen.flow(train_images_coarse,
                               train_labels_coarse,
                               batch_size=batch_size),
                  epochs=epochs,
                  validation_data=datagen.flow(test_images_coarse,
                                               test_labels_coarse),
                  shuffle=True,
                  callbacks=[WandbCallback(data_type='image',
                                           validation_data=(test_images_coarse,
                                                            test_labels_coarse),
                                           labels=class_names
                                           )
                             ]

                  )

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
    # model.add(layers.Dense(n_penul, activation=leaky_relu))
    model_2.add(layers.Dense(n_fine_class, activation='softmax'))

    print("summary after adding new layer")
    sgd_2 = tf.keras.optimizers.SGD(learning_rate=learning_rate_fine, momentum=.9)
    model_2.build(input_shape=model.input_shape)
    model_2.summary()
    model_2.compile(
        # optimizer=adam,
        optimizer=sgd_2,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model_2.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=(test_images, test_labels),
        callbacks=[WandbCallback(
            data_type='image',
            validation_data=(test_images,
                             test_labels)
            )
        ]
    )

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    test_loss, test_acc = model_2.evaluate(test_images, test_labels, verbose=2)

    print(test_acc)
    wandb.log({"novel_task_test_accuracy": test_acc})
    wandb.log({"training_accuracy_plot": plt})
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    #
    run_transfer_learning(
        transfer=True,
        batch_norm=False,
        dropout=False,
        penultimate_layer_dim=args.penultimate_layer_dim,
        epochs=1
    )
