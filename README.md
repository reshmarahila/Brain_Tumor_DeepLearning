=> Loads MRI images from train/, val/, and test/ folders using ImageDataGenerator.

=> Applies data augmentation (rotation, shift, flip, zoom) to training images for better generalization.

=> Handles class imbalance by computing class_weight so the model doesn't always favor the most common tumor.

=> Uses EfficientNetB0 as a pretrained base model (from ImageNet) with a custom classification head.

=> Trains the model using your labeled dataset (4 classes: Glioma, Meningioma, No Tumor, Pituitary).

=> Monitors validation loss and uses early stopping + checkpoint to save the best version.

=> Evaluates the trained model on the test set to calculate accuracy.

=> Generates a classification report (precision, recall, F1-score) for detected classes.

=> Plots a confusion matrix to visualize prediction correctness per class.

**What this code does — in a few points**
1. Jupyter Notebook (train_model.ipynb)
1. Loads the dataset: Reads brain MRI images from the train, val, and test folders.
2. Identifies four classes: Glioma, Meningioma, No Tumor, and Pituitary.
3. Preprocesses images: Resizes images to 224 × 224 pixels and applies data augmentation to training images.
4. Builds the model: Uses pretrained EfficientNetB0 to extract image features and classify MRI scans into four classes.
5. Trains the model: Learns to distinguish the tumor categories using labeled MRI images.
6. Evaluates performance: Measures test accuracy, generates a classification report, and creates a confusion matrix.
7. Saves the results: Stores the trained model as efficientnet_model.keras and the class labels as class_names.json.
2. Streamlit (app.py)
1. Loads the trained model and its corresponding class labels.
2. Provides a web interface for uploading an MRI image.
3. Processes the image to match the model's expected input size.
4. Predicts the class with the highest model score.
5. Displays results: Shows the predicted category, confidence score, and scores for all four classes.
6. Handles errors: Displays messages if files are missing or an image cannot be processed.
