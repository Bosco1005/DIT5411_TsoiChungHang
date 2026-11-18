# train.py
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Activation, GlobalAveragePooling2D, Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from config import *

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        "data/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        "data/test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )
    return train_gen, test_gen


def build_model_resnet_style(num_classes):
    """Residual-style CNN"""
    inputs = Input(shape=(*IMG_SIZE, 3))
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, 2)(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, 2)(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model


def build_model_efficient(num_classes):
    """EfficientNet-style lightweight model"""
    model = Sequential([
        Conv2D(24, (3, 3), padding='same', input_shape=(*IMG_SIZE, 3)),
        BatchNormalization(),
        Activation('swish'),
        MaxPooling2D(2, 2),
        
        Conv2D(40, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('swish'),
        MaxPooling2D(2, 2),
        
        Conv2D(80, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('swish'),
        Conv2D(80, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('swish'),
        MaxPooling2D(2, 2),
        
        Conv2D(112, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('swish'),
        
        GlobalAveragePooling2D(),
        Dense(256, activation='swish'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_and_evaluate(model, train_gen, test_gen, name):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"\n=== Training {name} ===")
    model.summary()
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        verbose=1
    )
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"{name} Test Accuracy: {test_acc:.4f}")
    model.save(f"{name}.h5")
    return test_acc, history.history


if __name__ == "__main__":
    train_gen, test_gen = get_generators()
    num_classes = train_gen.num_classes
    print(f"Number of classes: {num_classes}")

    results = {}

    # Model 1
    model1 = build_model_resnet_style(num_classes)
    acc1, _ = train_and_evaluate(model1, train_gen, test_gen, "model_resnet_style")
    results["model_resnet_style"] = acc1

    # Model 2
    model2 = build_model_efficient(num_classes)
    acc2, _ = train_and_evaluate(model2, train_gen, test_gen, "model_efficient")
    results["model_efficient"] = acc2

    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Final Results ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")