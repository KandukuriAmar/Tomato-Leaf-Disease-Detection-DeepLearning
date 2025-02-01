**Tomato Leaf Disease Detection** 
This project uses a Convolutional Neural Network (CNN) to detect diseases in tomato leaves. Upload an image of a leaf, and the model will analyze it to diagnose potential issues.

**Technologies used:** 
* Python
* TensorFlow/Keras
* Git & Git LFS (for handling the large .h5 model file)
* HTML/CSS (for the web interface)
  
**Project Structure** 
Tomato-Leaf-Disease-Detection-DeepLearning/
* TomatoDiseaseDetection.py         # Main script
* TomatoDiseaseDetection_optimizedvscode2nd.h5  # Trained CNN model (via Git LFS)
* templates/
*    index.html                    # Web interface
*    .gitattributes                    # Git LFS config
*     README.md                         # This file

**Clone the Repository:** 
git clone https://github.com/KandukuriAmar/Tomato-Leaf-Disease-Detection-DeepLearning.git

**Install Dependencies:** 
pip install -r requirements.txt

**Run the Application:** 
python TomatoDiseaseDetection.py
