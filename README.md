# Grape Leaf Disease Detection: CNN Feature Extraction with XGBoost

## Overview
This project replicates the study by Ishengoma and Lyima (Heliyon, 2024) on grape leaf disease detection using CNN-based feature extraction and a classifier. The original study used a Random Forest classifier with a dual-channel CNN architecture, achieving 95.34% accuracy. In this replication, I replaced Random Forest with XGBoost, modified the data preprocessing pipeline, and achieved a comparable accuracy of 95.2%.

The dataset used is the PlantVillage Grape dataset, which includes four classes: Black-rot, Esca-black-measles, Healthy, and Leaf-blight-isariopsis-leafspot. Features are extracted using VGG16 and Xception models, combined, and classified using XGBoost.

## Features
- **Preprocessing**: Background removal using HSV color space, intensity averaging with 3x3 blur, and bilateral filtering.
- **Feature Extraction**: VGG16 and Xception models for feature extraction.
- **Classification**: XGBoost with 100 estimators, max depth of 4, and a learning rate of 0.2.
- **Data Augmentation**: Applied to all classes using ImageDataGenerator with class-specific parameters.
- **Evaluation**: Achieved 95.2% accuracy on the test set, with detailed classification metrics and confusion matrix.

## Requirements
- Python 3.8+
- Libraries: `numpy`, `opencv-python`, `Pillow`, `scikit-learn`, `tensorflow`, `xgboost`, `joblib`, `matplotlib`, `seaborn`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
- **Source**: PlantVillage Grape dataset (not included in this repository).
- **Structure**: Place the dataset in a folder named `grape_leaf_dataset` with subfolders for each class (Black-rot, Esca-black-measles, Healthy, Leaf-blight-isariopsis-leafspot).
- **Preprocessing**: Images are resized to 128x128 pixels during training.

## Usage
1. **Prepare the dataset**: Ensure the `grape_leaf_dataset` folder is in the project root.
2. **Run the script**:
   ```bash
   python main.py
   ```
3. **Outputs**:
   - `xgb_model.pkl`: Trained XGBoost model.
   - `class_map.npy`: Mapping of class names to indices.
   - `train.txt`, `val.txt`, `test.txt`: Lists of image paths for each split.
   - Logs with classification report and confusion matrix.

## Results
- **Accuracy**: 95.2% on the test set (vs. 95.34% in the original study).
- **Classification Metrics**:
  - Black-rot: Precision 0.91, Recall 0.96, F1-score 0.93
  - Esca-black-measles: Precision 0.96, Recall 0.90, F1-score 0.93
  - Leaf-blight-isariopsis-leafspot: Precision 1.00, Recall 1.00, F1-score 1.00
  - Healthy: Precision 1.00, Recall 1.00, F1-score 1.00
- **Confusion Matrix**: Most errors occur between Black-rot and Esca-black-measles due to visual similarity.

## Differences from Original Study
- Used XGBoost instead of Random Forest for better handling of imbalanced data.
- Applied broader data augmentation across all classes.
- Reduced image size to 128x128 pixels (vs. 256x256 in the original) to lower memory usage.
- Simplified the dual-channel architecture by sequentially combining features instead of parallel processing.

## Future Improvements
- Increase image resolution to 200x200 or higher for better detail capture.
- Explore hybrid CNN-GNN architectures for improved feature extraction.
- Integrate attention mechanisms (e.g., CBAM) to focus on disease-affected areas.
- Use GANs for advanced data augmentation.

## Citation
This work replicates:
- Ishengoma, F. S., & Lyima, N. N. (2024). "Ensemble Model for Grape Leaf Disease Detection Using CNN Feature Extraction and Random Forest Classifier." *Heliyon*.

The paper has been accepted at the Fourth International Conference on AI ML Systems, October 8-11, 2024, Louisiana State University, Baton Rouge, USA.

## License
This project is licensed under the MIT License.