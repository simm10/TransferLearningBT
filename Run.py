import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD, Adadelta
from sklearn.metrics import confusion_matrix, classification_report

RESCALE = 1. / 255
DATASET_DIR = ""
if str(sys.argv[1]) == "binary":
    DATASET_DIR = "Change it"
elif str(sys.argv[1]) == "multiclass":
    DATASET_DIR = "Change it"
TEST_DATA_DIR = DATASET_DIR + "test"
VALID_DATA_DIR = DATASET_DIR + "valid"
TRAIN_DATA_DIR = DATASET_DIR + "train"
THIS_DIR = "Change it"
LOG_DIR = THIS_DIR + "/logs"
WEIGHTS_PATH = THIS_DIR + "/models/"
EPOCHS = 100

tensorboard = TensorBoard(log_dir=LOG_DIR,
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

check_pointer = ModelCheckpoint(filepath=WEIGHTS_PATH + "weights.h5",
                                verbose=1,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True)

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=10,
                           verbose=0,
                           mode='min')

callbacks_list = [check_pointer, tensorboard, early_stop]

SMALL_INPUT_DIMENSIONS = 150, 150, 3
BIG_BATCH_SIZE = 32
SMALL_BATCH_SIZE = 16
BIG_INPUT_DIMENSIONS = 224, 224, 3
BINARY_GENERATOR_CLASS_MODE = "binary"
MULTICLASS_GENERATOR_CLASS_MODE = "categorical"


def get_binary_model():
    model = Sequential()

    model.add(Convolution2D(filters=32, kernel_size=(3, 3),
                            input_shape=SMALL_INPUT_DIMENSIONS,
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.003, momentum=0.9),
                  metrics=['accuracy'])

    model.summary()

    # sys.exit()

    return model


def train():
    model = get_binary_model()

    datagen = ImageDataGenerator(rescale=RESCALE,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

    train_generator = datagen.flow_from_directory(TRAIN_DATA_DIR,
                                                  target_size=(SMALL_INPUT_DIMENSIONS[0], SMALL_INPUT_DIMENSIONS[1]),
                                                  batch_size=BIG_BATCH_SIZE,
                                                  class_mode=BINARY_GENERATOR_CLASS_MODE,
                                                  shuffle=True)

    valid_generator = datagen.flow_from_directory(VALID_DATA_DIR,
                                                  target_size=(SMALL_INPUT_DIMENSIONS[0], SMALL_INPUT_DIMENSIONS[1]),
                                                  batch_size=BIG_BATCH_SIZE,
                                                  class_mode=BINARY_GENERATOR_CLASS_MODE,
                                                  shuffle=True)

    validation_samples = valid_generator.n

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.n // BIG_BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=valid_generator,
                        validation_steps=validation_samples // BIG_BATCH_SIZE,
                        callbacks=callbacks_list,
                        verbose=1,
                        workers=12)
    # sys.exit()
    score = model.evaluate_generator(valid_generator, validation_samples, workers=12)

    print('Validačná správnosť:', (score[1] * 100.0), "%")

    sys.exit()


def test():
    datagen = ImageDataGenerator(rescale=RESCALE)

    test_generator = datagen.flow_from_directory(TEST_DATA_DIR,
                                                 target_size=(SMALL_INPUT_DIMENSIONS[0], SMALL_INPUT_DIMENSIONS[1]),
                                                 batch_size=BIG_BATCH_SIZE,
                                                 class_mode=BINARY_GENERATOR_CLASS_MODE,
                                                 shuffle=False)

    model = get_binary_model()

    model.load_weights(filepath=WEIGHTS_PATH + "binaryWeights.h5")

    probabilities = model.predict_generator(test_generator,
                                            steps=test_generator.n // BIG_BATCH_SIZE + 1,
                                            workers=12,
                                            verbose=1)

    test_samples = len(test_generator.filenames)

    y_true = np.array([0] * int(test_samples / 2) + [1] * int(test_samples / 2))
    y_pred = probabilities > 0.5

    matrix = confusion_matrix(y_true, y_pred)

    TN = (matrix[0][0])
    FN = (matrix[1][0])
    FP = (matrix[0][1])
    TP = (matrix[1][1])
    print("Matica zámien")
    print("TN:", TN, "\tFP:", FP)
    print("FN:", FN, "\tTP:", TP)

    print("Správnosť:", (TP + TN) / test_samples * 100, "%")
    print("Misklasifikácia:", (1 - ((TP + TN) / test_samples)) * 100, "%")


def get_binary_model_tl():
    vgg_model = VGG16(include_top=False, weights='imagenet',
                      input_shape=BIG_INPUT_DIMENSIONS)

    for layer in vgg_model.layers[:-4]:
        layer.trainable = False

    model = Sequential()

    for layer in vgg_model.layers:
        model.add(layer)

    model.add(Flatten())

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.001, decay=0.00001, momentum=0.9),
                  metrics=['accuracy'])

    model.summary()

    # sys.exit()

    return model


def train_tl():
    datagen = ImageDataGenerator(rescale=RESCALE,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

    train_generator = datagen.flow_from_directory(TRAIN_DATA_DIR,
                                                  target_size=(BIG_INPUT_DIMENSIONS[0], BIG_INPUT_DIMENSIONS[1]),
                                                  batch_size=SMALL_BATCH_SIZE,
                                                  class_mode=BINARY_GENERATOR_CLASS_MODE,
                                                  shuffle=True)

    valid_generator = datagen.flow_from_directory(VALID_DATA_DIR,
                                                  target_size=(BIG_INPUT_DIMENSIONS[0], BIG_INPUT_DIMENSIONS[1]),
                                                  batch_size=SMALL_BATCH_SIZE,
                                                  class_mode=BINARY_GENERATOR_CLASS_MODE,
                                                  shuffle=True)

    validation_samples = valid_generator.n

    model = get_binary_model_tl()

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.n // SMALL_BATCH_SIZE,
                        epochs=100,
                        validation_data=valid_generator,
                        validation_steps=validation_samples // SMALL_BATCH_SIZE,
                        callbacks=callbacks_list,
                        verbose=1,
                        workers=12)
    # sys.exit()
    score = model.evaluate_generator(valid_generator,
                                     validation_samples,
                                     workers=12)

    print('Validačná správnosť:', (score[1] * 100.0), "%")

    sys.exit()


def test_tl():
    model = get_binary_model_tl()

    model.load_weights(filepath=WEIGHTS_PATH + "binaryWeightsTl.h5")

    datagen = ImageDataGenerator(rescale=RESCALE)

    test_generator = datagen.flow_from_directory(TEST_DATA_DIR,
                                                 target_size=(BIG_INPUT_DIMENSIONS[0], BIG_INPUT_DIMENSIONS[1]),
                                                 batch_size=SMALL_BATCH_SIZE,
                                                 class_mode=BINARY_GENERATOR_CLASS_MODE,
                                                 shuffle=False)

    probabilities = model.predict_generator(test_generator,
                                            steps=test_generator.n // SMALL_BATCH_SIZE + 1,
                                            workers=12,
                                            verbose=1)

    test_samples = len(test_generator.filenames)

    y_true = np.array([0] * int(test_samples / 2) + [1] * int(test_samples / 2))
    y_pred = probabilities > 0.5

    matrix = confusion_matrix(y_true, y_pred)

    TN = (matrix[0][0])
    FN = (matrix[1][0])
    FP = (matrix[0][1])
    TP = (matrix[1][1])
    print("TN:", TN, "\tFP:", FP)
    print("FN:", FN, "\tTP:", TP)

    print("Správnosť:", (TP + TN) / test_samples * 100, "%")
    print("Misklasifikácia:", (1 - ((TP + TN) / test_samples)) * 100, "%")


def get_multiclass_model():
    model = Sequential()

    model.add(Convolution2D(64, (3, 3), input_shape=SMALL_INPUT_DIMENSIONS,
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(101, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, decay=0.000001, momentum=0.9),
                  metrics=['accuracy'])

    model.summary()

    # sys.exit()

    return model


def train_multiclass():
    datagen = ImageDataGenerator(rescale=RESCALE,
                                 zoom_range=[0.9, 1.1],
                                 brightness_range=[0.9, 1.1],
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)

    train_generator = datagen.flow_from_directory(TRAIN_DATA_DIR,
                                                  target_size=(SMALL_INPUT_DIMENSIONS[0], SMALL_INPUT_DIMENSIONS[1]),
                                                  batch_size=BIG_BATCH_SIZE,
                                                  class_mode=MULTICLASS_GENERATOR_CLASS_MODE,
                                                  shuffle=True)

    valid_generator = datagen.flow_from_directory(VALID_DATA_DIR,
                                                  target_size=(SMALL_INPUT_DIMENSIONS[0], SMALL_INPUT_DIMENSIONS[1]),
                                                  batch_size=BIG_BATCH_SIZE,
                                                  class_mode=MULTICLASS_GENERATOR_CLASS_MODE,
                                                  shuffle=True)

    validation_samples = valid_generator.n

    model = get_multiclass_model()

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.n // BIG_BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=valid_generator,
                        validation_steps=validation_samples // BIG_BATCH_SIZE,
                        callbacks=callbacks_list,
                        verbose=1,
                        workers=12)
    # sys.exit()
    score = model.evaluate_generator(valid_generator,
                                     validation_samples,
                                     workers=12)

    print('Validačná správnosť:', (score[1] * 100.0), "%")

    sys.exit()


def test_multiclass():
    model = get_multiclass_model()

    model.load_weights(filepath=WEIGHTS_PATH + "BB.h5")

    datagen = ImageDataGenerator(rescale=RESCALE, horizontal_flip=True)

    test_generator = datagen.flow_from_directory(TEST_DATA_DIR,
                                                 target_size=(SMALL_INPUT_DIMENSIONS[0], SMALL_INPUT_DIMENSIONS[1]),
                                                 batch_size=BIG_BATCH_SIZE,
                                                 class_mode=MULTICLASS_GENERATOR_CLASS_MODE,
                                                 shuffle=False)

    probabilities = model.predict_generator(test_generator,
                                            steps=test_generator.n // BIG_BATCH_SIZE + 1,
                                            workers=12,
                                            verbose=1)

    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(test_generator.classes, predictions)
    print(report)
    matrix = confusion_matrix(test_generator.classes, predictions)

    dataframe = pd.DataFrame(matrix,
                             index=[i for i in range(101)],
                             columns=[i for i in range(101)])
    plt.figure(figsize=(10, 7))
    plt.imshow(dataframe.div(dataframe.sum(axis=1), axis=0), cmap='plasma')
    plt.colorbar()
    plt.show()

    sys.exit()


def get_multiclass_model_tl():
    vgg_model = VGG16(include_top=False, weights='imagenet',
                      input_shape=BIG_INPUT_DIMENSIONS)

    model = Sequential()

    for layer in vgg_model.layers:
        model.add(layer)
        layer.trainable = False

    model.add(Flatten())

    model.add(Dropout(0.2))

    model.add(Dense(101, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(),
                  metrics=['accuracy'])

    model.summary()

    # sys.exit()

    return model


def train_multiclass_tl():
    datagen = ImageDataGenerator(rescale=RESCALE,
                                 zoom_range=[0.9, 1.1],
                                 brightness_range=[0.9, 1.1],
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)

    train_generator = datagen.flow_from_directory(TRAIN_DATA_DIR,
                                                  target_size=(BIG_INPUT_DIMENSIONS[0], BIG_INPUT_DIMENSIONS[1]),
                                                  batch_size=SMALL_BATCH_SIZE,
                                                  class_mode=MULTICLASS_GENERATOR_CLASS_MODE,
                                                  shuffle=True)

    valid_generator = datagen.flow_from_directory(VALID_DATA_DIR,
                                                  target_size=(BIG_INPUT_DIMENSIONS[0], BIG_INPUT_DIMENSIONS[1]),
                                                  batch_size=SMALL_BATCH_SIZE,
                                                  class_mode=MULTICLASS_GENERATOR_CLASS_MODE,
                                                  shuffle=True)

    validation_samples = valid_generator.n

    model = get_multiclass_model_tl()

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.n // SMALL_BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=valid_generator,
                        validation_steps=validation_samples // SMALL_BATCH_SIZE,
                        callbacks=callbacks_list,
                        verbose=1,
                        workers=12)
    # sys.exit()
    score = model.evaluate_generator(valid_generator,
                                     validation_samples,
                                     workers=12)

    print('Validačná spravnost:', (score[1] * 100.0), "%")

    sys.exit()


def test_multiclass_tl():
    model = get_multiclass_model_tl()
    model.load_weights(filepath=WEIGHTS_PATH + "MTLB.h5")

    datagen = ImageDataGenerator(rescale=RESCALE, horizontal_flip=True)

    test_generator = datagen.flow_from_directory(TEST_DATA_DIR,
                                                 target_size=(BIG_INPUT_DIMENSIONS[0], BIG_INPUT_DIMENSIONS[1]),
                                                 batch_size=SMALL_BATCH_SIZE,
                                                 class_mode=MULTICLASS_GENERATOR_CLASS_MODE,
                                                 shuffle=False)

    probabilities = model.predict_generator(test_generator,
                                            steps=test_generator.n // SMALL_BATCH_SIZE + 1,
                                            workers=12,
                                            verbose=1)

    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(test_generator.classes, predictions)
    print(report)
    matrix = confusion_matrix(test_generator.classes, predictions)

    dataframe = pd.DataFrame(matrix,
                             index=[i for i in range(101)],
                             columns=[i for i in range(101)])

    plt.figure(figsize=(10, 7))
    plt.imshow(dataframe.div(dataframe.sum(axis=1), axis=0), cmap='plasma')
    plt.colorbar()
    plt.show()

    sys.exit()


mode = str(sys.argv[1])
cmd = str(sys.argv[2])
if mode == "binary":
    if cmd == "train":
        print("Training")
        train()
    elif cmd == "test":
        test()
    elif cmd == "train_tl":
        train_tl()
    elif cmd == "test_tl":
        test_tl()
elif mode == "multiclass":
    if cmd == "train":
        train_multiclass()
    elif cmd == "test":
        test_multiclass()
    elif cmd == "train_tl":
        train_multiclass_tl()
    elif cmd == "test_tl":
        test_multiclass_tl()
else:
    print("Unknown mode")
    sys.exit()
