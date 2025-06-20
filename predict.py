import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from skimage.feature import hog

def preprocess_image(image_path):
    """Replicates preprocessing pipeline with channel fix"""
    # Load and check blurriness
    if cv2.Laplacian(cv2.imread(image_path, 0), cv2.CV_64F).var() < 10:
        print("Warning: Image might be too blurry for accurate prediction")

    # Full preprocessing chain
    img = cv2.imread(image_path)
    img = cv2.resize(img, (220, 220))
    
    # Histogram equalization
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    # Sobel filtering
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(sobel_combined)
    
    # Segmentation and channel fix
    _, thresh = cv2.threshold(sobel_combined, 60, 255, cv2.THRESH_BINARY_INV)
    final_img = cv2.resize(thresh, (64, 64))
    
    # Convert to 3-channel for CNN compatibility
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

def predict(image_path):
    """Main prediction function with shape correction"""
    # Load saved components
    cnn_model = load_model('cnn_feature_extractor.h5')
    svm_model = joblib.load('svm_classifier.joblib')
    le = joblib.load('label_encoder.joblib')
    
    # Preprocess and extract features
    processed_img = preprocess_image(image_path)
    
    # CNN features with proper 4D input (batch, height, width, channels)
    cnn_input = np.expand_dims(processed_img/255.0, axis=0)  # Now shape (1, 64, 64, 3)
    cnn_features = cnn_model.predict(cnn_input)
    
    # HOG features (convert back to grayscale for HOG)
    hog_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(hog_img, orientations=9, 
                      pixels_per_cell=(8,8), cells_per_block=(2,2))
    hog_features = np.expand_dims(hog_features, axis=0)
    
    # Combine and predict
    combined_features = np.hstack((cnn_features, hog_features))
    pred = svm_model.predict(combined_features)
    
    return le.inverse_transform(pred)[0]

# print("Predicted:", predict('Images/sample4.jpg'))
