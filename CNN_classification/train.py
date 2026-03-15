import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

IMG_SIZE = 224
BATCH_SIZE = 32

train_dir = "dataset/train"
val_dir = "dataset/val"

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

model = build_model(IMG_SIZE)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8
)

model.save("attention_cnn.h5")

print("Model trained and saved.")
