import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import shutil
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# ** CHANGE THESE TO YOUR DATASET PATHS **
IMAGES_FOLDER = r'/Users/namburunainavismi/Desktop/Sem5/Deep Learning/COVID-19_Radiography_Dataset'
LUNG_OPACITY_FOLDER = r'/Users/namburunainavismi/Desktop/Sem5/Deep Learning/COVID-19_Radiography_Dataset'

OUTPUT_FOLDER = './results_efficientnet'
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("EFFICIENTNETB0 - COVID-19 BIAS DETECTION MODEL")
print("="*70)


# ==================== SETUP ====================
def setup_folders():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs('./prepared_data', exist_ok=True)
    print(f"✓ Output folder created: {OUTPUT_FOLDER}\n")


# ==================== DATA LOADING ====================
def load_data_from_folders(images_folders):
    print("📁 Loading images from folders...")
    all_data = []
    total_images = 0

    for folder in images_folders:
        print(f"   Loading from: {folder}")
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder} - skipping")
            continue

        image_files = [f for f in os.listdir(folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

        if len(image_files) == 0:
            print(f"⚠️  No images found in {folder} - skipping")
            continue

        print(f"   ✓ Found {len(image_files)} images")
        total_images += len(image_files)

        data = []
        failed = []

        for filename in image_files:
            try:
                base = filename.rsplit('.', 1)[0]
                if '-' in base:
                    label = base.split('-')[0]
                elif '_' in base:
                    label = base.split('_')[0]
                else:
                    label = ''.join([c for c in base if c.isalpha()])
                if label:
                    data.append({
                        'filename': filename,
                        'filepath': os.path.join(folder, filename),
                        'label': label.upper()
                    })
                else:
                    failed.append(filename)
            except Exception:
                failed.append(filename)

        if failed:
            print(f"   ⚠️  Could not extract labels from {len(failed)} files")
        all_data.extend(data)

    if total_images == 0:
        raise ValueError("❌ No images found in any specified folders")

    df = pd.DataFrame(all_data)
    classes = sorted(df['label'].unique())
    print(f"✓ Loaded {len(df)} images with {len(classes)} classes: {classes}\n")

    print("📊 Class Distribution:")
    for cls in classes:
        count = len(df[df['label'] == cls])
        print(f"   {cls}: {count} images ({count/len(df)*100:.1f}%)")
    print()
    return df, classes


# ==================== DATA PREPARATION ====================
def prepare_dataset_structure(df, classes, val_split=0.2, test_split=0.1):
    print("🔨 Preparing dataset structure...")
    base_dir = './prepared_data'

    if os.path.exists(base_dir):
        print("⚠️  Removing existing prepared_data directory...")
        try:
            if os.name == 'nt':
                os.system(f'rd /s /q "{base_dir}"')
            else:
                shutil.rmtree(base_dir)
        except Exception as e:
            print(f"Error removing old data: {e}")
            raise

    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

    for cls in classes:
        cls_df = df[df['label'] == cls].reset_index(drop=True)
        train_val_idx, test_idx = train_test_split(
            cls_df.index, test_size=test_split, random_state=42)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_split/(1-test_split), random_state=42)

        for idx in train_idx:
            shutil.copy2(cls_df.loc[idx, 'filepath'],
                         os.path.join(base_dir, 'train', cls, cls_df.loc[idx, 'filename']))
        for idx in val_idx:
            shutil.copy2(cls_df.loc[idx, 'filepath'],
                         os.path.join(base_dir, 'val', cls, cls_df.loc[idx, 'filename']))
        for idx in test_idx:
            shutil.copy2(cls_df.loc[idx, 'filepath'],
                         os.path.join(base_dir, 'test', cls, cls_df.loc[idx, 'filename']))

        print(f"   {cls}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    print(f"\n✓ Dataset prepared at: {base_dir}\n")
    return base_dir


# ==================== DATA GENERATORS ====================
def create_data_generators(base_dir, img_height, img_width, batch_size):
    print("🔄 Creating data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb')
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(base_dir, 'val'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb')
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb')

    print(f"✓ Generators ready! Classes: {train_gen.class_indices}\n")
    return train_gen, val_gen, test_gen


# ==================== MODEL BUILDING ====================
def build_efficientnetb0_model(num_classes, img_height, img_width, learning_rate):
    print("🏗️  Building EfficientNetB0 model...")

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    print(f"✓ Model built with {model.count_params():,} total parameters\n")
    return model


# ==================== TRAINING ====================
def train_model(model, train_gen, val_gen, epochs):
    print("🚀 Training started...\n")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_FOLDER, 'best_model_efficientnet.keras'),
            monitor='val_auc', mode='max', save_best_only=True)
    ]
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    print("✓ Training complete!\n")
    return history


# ==================== VISUALIZATION ====================
def plot_training_history(history):
    print("📊 Plotting training history...")
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['accuracy', 'loss', 'auc']
    for i, metric in enumerate(metrics):
        ax[i//2, i % 2].plot(history.history[metric], label='Train', marker='o')
        ax[i//2, i % 2].plot(history.history['val_' + metric], label='Validation', marker='s')
        ax[i//2, i % 2].set_title(f'Model {metric.capitalize()}')
        ax[i//2, i % 2].legend()
        ax[i//2, i % 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'training_history_efficientnet.png'), dpi=300)
    plt.close()


# ==================== EVALUATION ====================
def evaluate_model(model, test_gen, class_names):
    print("📈 Evaluating model...")
    predictions = model.predict(test_gen)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes

    report = classification_report(true_classes, pred_classes, target_names=class_names)
    print(report)
    with open(os.path.join(OUTPUT_FOLDER, 'classification_report_efficientnet.txt'), 'w') as f:
        f.write(report)

    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title("Confusion Matrix - EfficientNetB0")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix_efficientnet.png'), dpi=300)
    plt.close()


# ==================== MAIN ====================
def main():
    setup_folders()
    print("TensorFlow version:", tf._version_)
    print("Keras version:", keras._version_)
    df, classes = load_data_from_folders([IMAGES_FOLDER, LUNG_OPACITY_FOLDER])
    base_dir = prepare_dataset_structure(df, classes, VALIDATION_SPLIT, TEST_SPLIT)
    train_gen, val_gen, test_gen = create_data_generators(base_dir, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
    # Debug: print shape of a sample batch to confirm channels
    print("Sample batch shape (should end with 3):", train_gen[0][0].shape)
    try:
        model = build_efficientnetb0_model(len(classes), IMG_HEIGHT, IMG_WIDTH, LEARNING_RATE)
    except ValueError as e:
        print("\n[WARNING] Could not load imagenet weights due to shape mismatch. Building model with random weights.\nError:", e)
        def build_efficientnetb0_model_no_weights(num_classes, img_height, img_width, learning_rate):
            print("🏗️  Building EfficientNetB0 model (no pre-trained weights)...")
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=(img_height, img_width, 3)
            )
            base_model.trainable = False
            inputs = keras.Input(shape=(img_height, img_width, 3))
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            model = keras.Model(inputs, outputs)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
            )
            print(f"✓ Model built with {model.count_params():,} total parameters\n")
            return model
        model = build_efficientnetb0_model_no_weights(len(classes), IMG_HEIGHT, IMG_WIDTH, LEARNING_RATE)
    history = train_model(model, train_gen, val_gen, EPOCHS)
    plot_training_history(history)
    evaluate_model(model, test_gen, classes)
    model.save(os.path.join(OUTPUT_FOLDER, 'final_model_efficientnet.keras'))
    print("\n✅ All results saved successfully in", OUTPUT_FOLDER)


if _name_ == "_main_":
    main()