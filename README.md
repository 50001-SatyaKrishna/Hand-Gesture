# Hand-Gesture
# Hand Gesture Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** in TensorFlow/Keras to classify hand gestures into multiple categories.  
The trained model can recognize hand gestures from images with high accuracy.

---

## 🚀 Features
- Image classification using CNN.
- Data preprocessing and augmentation with `ImageDataGenerator`.
- Achieved **~98% validation accuracy** on the dataset.
- Saved trained model (`hand_gesture_model.h5`) for reuse.
- Supports single-image prediction with visualization.

---

## 📂 Project Structure
├── app.py # Main script (optional for deployment/Streamlit)
├── Stock Predictions Model.keras
├── Stockmarketprediction.ipynb
├── hand_gesture_model.h5 # Trained CNN model
├── train_03/ # Training dataset (10 classes)
├── test_03/ # Testing dataset
└── README.md

yaml
Copy
Edit

---

## 📊 Dataset
- **Training set**: 1600 images belonging to 10 classes  
- **Validation set**: 400 images belonging to 10 classes  
- Images resized to **128x128** pixels.  
- Example classes: `Ok sign`, `Thumbs up`, `Stop`, etc.

---

## 🧑‍💻 Model Architecture
- **Conv2D + MaxPooling** layers (32, 64, 128 filters)  
- **Flatten layer**  
- **Dense layer (512 units, ReLU)**  
- **Output layer (Softmax)**  

Optimizer: **Adam**  
Loss: **Categorical Crossentropy**

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/Stock-Market-Price-Prediction.git
cd Stock-Market-Price-Prediction

pip install -r requirements.txt
Requirements
numpy

pandas

matplotlib

scikit-learn

tensorflow

keras

streamlit

yfinance (if stock prediction module is used)

🏋️ Training
To train the model, run:

bash
Copy
Edit
python train.py
(or run cells in Jupyter Notebook)

🔮 Prediction
Use the saved model to predict on a new image:

python
Copy
Edit
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("hand_gesture_model.h5")

img = image.load_img("test_03/frame_02_07_0009.png", target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
print("Predicted Gesture:", predicted_class)
📈 Results
Training Accuracy: ~100%

Validation Accuracy: ~98%

Example Prediction: "The predicted hand gesture is: Ok sign"

📌 Future Work
Deploy model with Streamlit web app.

Expand dataset for more gesture categories.

Improve real-time predictions with webcam input.

