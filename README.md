# Brain_Tumor_DeepLearning

** Brain Tumor Classifier – Code Summary**

1.📁 Loads and Prepares Dataset

Loads MRI images from train/, val/, and test/ folders.

Applies augmentation (flip, zoom, shift) to training data.

Uses sparse labels (integer values for classes).

2.🔄 Uses Transfer Learning (EfficientNetB0)

Loads a pretrained EfficientNetB0 model (without its top layers).

Freezes the pretrained layers.

Adds a custom classification head for 4 tumor classes.

3.⚙️ Compiles the Model

Loss function: sparse_categorical_crossentropy

Optimizer: Adam

Metric: accuracy

4.🏋️ Trains the Model

Runs training on the dataset for up to 30 epochs.

Uses EarlyStopping to avoid overfitting.

Saves the best model using ModelCheckpoint.

5.✅ Evaluates Performance

Loads the best saved model.

Evaluates on the test set.

Calculates accuracy and confusion matrix.

6.📊 Visualizes Results

Prints precision, recall, and F1-score per class.

Displays a labeled confusion matrix heatmap.
