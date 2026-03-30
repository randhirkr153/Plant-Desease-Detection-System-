import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import json

# Configuration
def get_path(path):
    # Convert Windows C:\ to WSL /mnt/c/ if running in WSL
    if os.name != 'nt' and path.lower().startswith('c:'):
        return path.replace('c:', '/mnt/c').replace('\\', '/')
    return path

DATASET_PATH = get_path(r"c:\Users\utsav\Desktop\project\archiveraak\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)")
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")
MODEL_SAVE_PATH = get_path(r"c:\Users\utsav\Desktop\project\plant_disease_flutter\assets\plant_disease_efficientnet.h5")
TFLITE_SAVE_PATH = get_path(r"c:\Users\utsav\Desktop\project\plant_disease_flutter\assets\model.tflite")
LABELS_PATH = get_path(r"c:\Users\utsav\Desktop\project\plant_disease_flutter\assets\labels.txt")
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS = 20

def clean_label(label):
    # Convert 'Apple___Apple_scab' to 'apple apple scab'
    return label.replace("___", " ").replace("_", " ").lower()

def train():
    print("Initializing Data Generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    labels = sorted(train_generator.class_indices.keys())
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        for label in labels:
            f.write(clean_label(label) + "\n")
        f.write("background\n")
    
    print(f"Labels saved to {LABELS_PATH}")

    # RESUME LOGIC
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Found existing model at {MODEL_SAVE_PATH}. Loading to resume...")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        # Check if we already finished initial training (if more than 50 layers are trainable)
        trainable_layers = sum([1 for l in model.layers if l.trainable])
        is_fine_tuning = trainable_layers > 10 
    else:
        print("Building EfficientNet-B0 Model from scratch...")
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(labels), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        is_fine_tuning = False

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]

    if not is_fine_tuning:
        print("Starting/Resuming Training (Initial Phase - Frozen Base)...")
        model.fit(
            train_generator,
            epochs=5,
            validation_data=valid_generator,
            callbacks=callbacks
        )
        
        print("Switching to Fine-tuning Model (Unfrozen Base)...")
        # Find the base model inside the functional model
        for layer in model.layers:
            if 'efficientnet' in layer.name:
                for sub_layer in layer.layers[-20:]:
                    sub_layer.trainable = True
                break
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Starting/Resuming Fine-tuning...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator,
        callbacks=callbacks
    )

    print(f"Saving final model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {TFLITE_SAVE_PATH}")

if __name__ == "__main__":
    train()
