# 📧 Spam Detection using Neural Network

A beginner‑friendly Machine Learning project that classifies emails as **Spam** or **Ham (Not Spam)** using a Neural Network with a logistic activation function.

This repository demonstrates the complete workflow of a traditional ML text classification system:

➡ Data → Preprocessing → Training → Prediction → Output file

It is designed mainly for students and first‑time contributors who want to understand how ML models were implemented before heavy frameworks became common.

---

## ✨ Features

* Text based spam classification
* Neural network implemented from scratch (no deep learning frameworks)
* CSV dataset training and prediction
* Simple reproducible workflow
* Beginner friendly code structure

---

## 🧠 How it Works

1. Reads labelled email dataset (Train.csv)
2. Converts text into numerical representation
3. Trains a neural network classifier
4. Predicts labels for unseen emails (TestX.csv)
5. Saves predictions into `output.csv`

---

## 📁 Project Structure

```
Spam-Detection/
│── Train.csv          Training dataset with labels
│── TestX.csv          Unseen emails for prediction
│── output.csv         Generated predictions
│── 120050018_2.py     Main ML model
│── README.md
│── LICENSE
```

---

## ⚙️ Requirements

Make sure you have:

* Python 3.8 or newer
* pip package manager

Install required libraries:

```bash
pip install numpy pandas
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Meenapintu/Spam-Detection.git
cd Spam-Detection
```

### 2. Install dependencies

```bash
pip install numpy pandas
```

### 3. Run the program

```bash
python 120050018_2.py
```

---

## 📊 Dataset Details

### Train.csv

Contains labelled emails used for learning.

| EmailText                     | Label |
| ----------------------------- | ----- |
| Congratulations you won prize | spam  |
| Meeting scheduled tomorrow    | ham   |

---

### TestX.csv

Contains emails without labels — the model predicts them.

| EmailText                 |
| ------------------------- |
| Free coupon available now |

---

### output.csv

Generated after running the script.

| Prediction |
| ---------- |
| spam       |
| ham        |

---

## 🖥️ Program Output

After successful execution you should see something like:

```
Training completed successfully
Predictions written to output.csv
```

---

## 📚 Learning Value

This project helps you understand:

* Text preprocessing in ML
* Feature extraction basics
* Neural network training logic
* Logistic activation function usage
* Model prediction pipeline

---

## 🛠 Possible Improvements

You can extend this project by adding:

* Better text vectorization (TF‑IDF)
* Accuracy metrics (precision/recall/F1)
* Confusion matrix visualization
* scikit‑learn implementation comparison
* Deep learning version (PyTorch/TensorFlow)
* Web interface for predictions

---

## 🤝 Contributing

Contributions are welcome!

You may improve:

* Documentation
* Code readability
* Performance
* Model accuracy

Steps:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a Pull Request

---

## 📜 License

This project is licensed under the MIT License.
