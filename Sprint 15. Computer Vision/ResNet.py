import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def load_train(path):
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rescale=1/255)
    train_data = datagen.flow_from_directory(path, target_size=(150, 150), class_mode='sparse', batch_size=16, seed=42)
    return train_data

def create_model(input_shape=(150, 150, 3), classes=12):
    backbone = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(classes, activation='softmax'))

    optimizer = Adam(lr=0.001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['acc'],
    )

    return model

def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=3,
    steps_per_epoch=None,
    validation_steps=None,
):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
    )

    return model
