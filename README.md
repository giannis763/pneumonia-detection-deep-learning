# Pneumonia Detection via Chest X-Rays using Deep Learning & Grad-CAM

This project was developed as part of my undergraduate thesis in Computer Science. The goal was to build a practical AI tool to assist in detecting pneumonia from chest X-ray images, while providing visual explanations for each diagnosis.

### Why I built it
In medical deep learning applications, it is not enough for a model to simply output whether an X-ray shows signs of disease. Clinicians need to understand why a specific decision was reached. To address this, I integrated Grad-CAM, which produces heatmaps overlaid on the input image to highlight the suspicious regions that drove the prediction.

### Key features
* Deep Learning Model: Trained neural networks using TensorFlow / Keras on an open-source chest X-ray dataset.
* Explainability (Grad-CAM): Visualized the image features and regions influencing the model's output.
* Web Application: An interactive Python web interface for uploading images and viewing predictions alongside heatmaps.
* Documentation: Includes the literature review, implementation methodology, and the complete thesis text.

### Repository structure
* App/ — Web application code (app.py), the trained model weights (best_final.keras), and requirements.
* Code/ — Jupyter Notebook (Thesis.ipynb) covering data preprocessing, model training, evaluation, and Grad-CAM generation.
* Dissertation/ — Complete thesis documentation and reports.

### Getting started

1. Clone the repository and navigate to the app directory:
   git clone https://github.com/giannis763/pneumonia-detection-deep-learning.git
   cd pneumonia-detection-deep-learning/App

2. Install the required dependencies:
   pip install -r requirements.txt

3. Run the application:
   python app.py
