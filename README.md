=> Loads MRI images from train/, val/, and test/ folders using ImageDataGenerator.

=> Applies data augmentation (rotation, shift, flip, zoom) to training images for better generalization.

=> Handles class imbalance by computing class_weight so the model doesn't always favor the most common tumor.

=> Uses EfficientNetB0 as a pretrained base model (from ImageNet) with a custom classification head.

=> Trains the model using your labeled dataset (4 classes: Glioma, Meningioma, No Tumor, Pituitary).

=> Monitors validation loss and uses early stopping + checkpoint to save the best version.

=> Evaluates the trained model on the test set to calculate accuracy.

=> Generates a classification report (precision, recall, F1-score) for detected classes.

=> Plots a confusion matrix to visualize prediction correctness per class.
