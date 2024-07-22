import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os 
import cv2                        #image processing

def read_files(rootdir):
    labels=[]          # list to append labels
    filepaths=[]        # list to append image paths
    #iterate through directory
    for root, subFolders, files in os.walk(rootdir):
        #iterate files in directory
        for file in files:
            #get filenames
            file_name = root + "/" + file
            filepaths.append(file_name)
            try:
                #get images and labels and append to the lists
                label = file_name.split("\\")[-1]
                label=label.split("/")[0]
                labels.append(label)          
            except Exception as e:
                print(e)

    return labels, filepaths


def plot_bar(data, label, count):
    # Plotting the bar plot
    plt.figure(figsize=(12, 9))
    # Plot the histogram
    plt.subplot(2, 1, 1)
    plt.bar(label, count, color='skyblue')
    plt.xlabel(' ')
    plt.ylabel('Count')
    plt.title('Count of Total Files in Each Class')
    plt.xticks(label)
    
    for i,lab in enumerate(label):
        img_path=data[data['Label']==lab]['Original_Image'].iloc[0]
        img=cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
        plt.subplot(2, len(label), len(label) + i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig("distribution.jpg")
    plt.tight_layout()
    plt.show()

def create_gen(train_data, val_data, test_data):
    # ImageDataGenerator for training with augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # ImageDataGenerator for validation (without augmentation)
    val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input
    )

    # ImageDataGenerator for testing (without augmentation)
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input
    )

    # Creating train, validation, and test image flows from dataframes
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_data,
        x_col='Original_Image',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
    )

    val_images = val_generator.flow_from_dataframe(
        dataframe=val_data,
        x_col='Original_Image',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_data,
        x_col='Original_Image',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    return train_generator, train_images, val_images, test_images








# Function to normalize images
def normalize_image(image):
    return (image - 127.5) / 127.5

# Function to reverse normalization
def reverse_normalize_image(image):
    return (image * 127.5) + 127.5

def display_augmented_images(image, datagen, num_examples=9):
    plt.figure(figsize=(10, 10))

    # Normalize the original image
    normalized_image = normalize_image(image)

    # Expand dimensions to match the input shape of the generator (batch of 1)
    normalized_image_expanded = np.expand_dims(normalized_image, axis=0)

    # Display the original image (after normalizing back to original range)
    plt.subplot(3, 3, 1)
    plt.title('Original Image')
    original_image_clipped = np.clip(reverse_normalize_image(normalized_image), 0, 255).astype('uint8')
    plt.imshow(original_image_clipped)
    plt.axis('off')

    # Generate augmented images
    augmented_images = datagen.flow(normalized_image_expanded, batch_size=1)

    # Display augmented images
    for i in range(1, num_examples):
        augmented_image = next(augmented_images)[0]
        augmented_image_clipped = np.clip(reverse_normalize_image(augmented_image), 0, 255).astype('uint8')

        plt.subplot(3, 3, i + 1)
        plt.title(f'Augmented Image {i}')
        plt.imshow(augmented_image_clipped)
        plt.axis('off')

    plt.show()