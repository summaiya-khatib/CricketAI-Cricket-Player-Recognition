import cv2
from app.recognizer import preprocess_and_save_temp
image_path = 'abd.jpg'
temp_path = preprocess_and_save_temp(image_path, debug=True)
img_cv2 = cv2.imread(temp_path)

print(f"CV2 loaded image shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
