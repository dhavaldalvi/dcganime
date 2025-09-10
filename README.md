# 🎨 DCGAN on Anime Face Dataset
## Anime face generator using DCGAN (Deep Convolutional GAN) implemented in TensorFlow.

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on an **Anime Face Dataset** to generate anime-style face images.

Note: This model trained with only 100 epochs. 

## 🎥 Progession of the training

  ![Image](https://github.com/user-attachments/assets/0fa13e36-475c-4b26-a165-d8ee4d4e4a20)
  
---

## 🧠 Model Overview

A **DCGAN** is a type of Generative Adversarial Network (GAN) that uses deep convolutional layers for both the Generator and Discriminator. The Generator learns to produce realistic anime faces, while the Discriminator learns to distinguish real images from generated ones.

---

## 📁 Dataset

The model is trained on the **Anime Face Dataset** consisting of cropped and aligned anime character faces.  

- 📦 Format: `.jpg` or `.png`
- 🔗 Source: https://www.kaggle.com/datasets/splcher/animefacedataset

---

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://https://github.com/dhavaldalvi/dcganime.git
cd dcganime
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place your anime face images inside the data/ directory.

### 4. Train the model

```bash
python main.py
```
The values of batch size and epoch are stored in dcgan/config/constants.py.

Or

### 4. Can use the saved model

you can use the saved models at "https://github.com/dhavaldalvi/dcganime/tree/main/outputs/models"


## 📌 Requirements
- Python 3.8-3.11
- tensorflow
- matplotlib
- opencv

## 🙌 Acknowledgments

- Thanks to the open-source community for tools and datasets.
- Special thanks to the creators of the Anime Face dataset.
