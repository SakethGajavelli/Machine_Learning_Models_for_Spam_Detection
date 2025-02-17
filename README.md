# 📊🚀 Spam Detection using NLP and Machine Learning Models ⛒📱

## Classifying SMS Messages with Naive Bayes, Logistic Regression, and Random Forest

### 🔍 Overview
In today’s digital age, unsolicited spam messages are a growing nuisance, leading to privacy concerns and productivity loss. This project leverages **Natural Language Processing (NLP)** and **Machine Learning** to classify SMS messages as either **Spam** or **Ham (Not Spam)**. Using the popular **SMS Spam Collection dataset**, this project implements and compares three powerful models:

- 🔍 **Naive Bayes** – Efficient for text classification with high accuracy.
- ⚖️ **Logistic Regression** – Provides interpretability with robust binary classification.
- 🌲 **Random Forest** – An ensemble method providing high accuracy and generalization.

---

## 🎯 Objectives
- Preprocess SMS text data using NLP techniques.
- Extract meaningful features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
- Implement and compare the performance of three machine learning models:
  - 🔍 Naive Bayes
  - ⚖️ Logistic Regression
  - 🌲 Random Forest
- Evaluate model performance using:
  - Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC Curves.
- Provide a **comparative analysis** and recommend the best model for SMS spam detection.

---

## 💽 Dataset
The project uses the **SMS Spam Collection dataset**, containing **5,572 SMS messages** labeled as either **Spam** or **Ham (Not Spam)**. It is publicly available on **Kaggle**:

- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets)

---

## 🛠️ Tech Stack and Tools Used
- **Programming Language:** Python
- **Libraries:**
  - **NLP and ML:** scikit-learn, numpy, pandas
  - **Visualization:** matplotlib, seaborn
- **Development Environment:** Google Colab

---

## ⚙️ Installation and Setup
### Clone the repository:
```sh
git clone https://github.com/your-username/spam-detection-nlp.git
cd spam-detection-nlp
```
### Install required packages:
```sh
pip install -r requirements.txt
```
### Download the Dataset:
1. Download the **SMS Spam Collection** dataset from Kaggle.
2. Place the dataset in the project directory.

### Run the Notebook:
- Open the project in **Google Colab** or **Jupyter Notebook**.
- Run the cells step by step to see the implementation and results.

---

## 📊 Results and Analysis
The models achieved the following accuracies:

| Model              | Accuracy |
|--------------------|----------|
| 🔍 **Naive Bayes**        | 98%      |
| ⚖️ **Logistic Regression** | 95%      |
| 🌲 **Random Forest**       | 98%      |

### Visualizations:
- 📊 **Confusion Matrices**: Visualized classification results.
- 📈 **ROC Curves**: Compared model performance across different thresholds.
- 📊 **Accuracy Comparison Bar Plot**: Displayed the overall accuracy of all models.

---

## ⚖️ Comparative Analysis
| Model | Strengths | Weaknesses |
|--------|------------|--------------|
| **Naive Bayes** | Fast, efficient, and high accuracy for short text. | Independence assumption can limit performance on complex datasets. |
| **Logistic Regression** | Good interpretability and balanced precision-recall. | Assumes a linear decision boundary, which may not capture complex patterns. |
| **Random Forest** | High accuracy and robustness through ensemble learning. | Computationally expensive and less interpretable. |

---

## 🚀 Future Directions
### **Advanced NLP Techniques:**
- Implement **Word2Vec**, **FastText**, or **Transformer-based models (BERT)** for richer feature representations.

### **Deep Learning Models:**
- Explore **LSTM (Long Short-Term Memory)** or **BiLSTM** models for sequential dependencies.

### **Generalization Testing:**
- Test on other datasets (e.g., email or social media spam) for robustness.

---

## 📝 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgements
- **UCI Machine Learning Repository** for the **SMS Spam Collection Dataset**.
- **Kaggle** for providing a platform to access the dataset.
- **scikit-learn** for the machine learning algorithms and tools.
- **Google Colab** for the cloud-based notebook environment.

---

## 📧 Contact
For questions or collaborations:
- **Your Name** – **Saketh Gajavelli**
- **Email:** sakethmunna220@gmail.com

---

## ⭐ Contributing
Contributions are welcome! Feel free to **open an issue** or **submit a pull request** for any feature suggestions or bug fixes.

If you find this project helpful, please **give it a star!** ⭐

