# 📧 Spam Email Detection (ML)

This project detects whether a given email/message is **Spam** or **Ham** using Machine Learning.

## 🚀 Tech Stack
- Python
- scikit-learn
- TF-IDF Vectorizer
- Naive Bayes Classifier

## 📂 Project Structure
- `data/` : dataset
- `src/` : training & prediction scripts
- `notebooks/` : Jupyter notebooks for EDA
- `requirements.txt` : dependencies

- ## 📂 Dataset
Download the dataset from [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
Place it inside the `data/` folder as `spam.csv`.


## 🔧 How to Run
```bash
git clone https://github.com/<your-username>/spam-email-detection.git
cd spam-email-detection
pip install -r requirements.txt
python src/train_model.py
python src/predict.py
```

## 📊 Results
- Accuracy ~97%
- Works on unseen test messages

## 📌 Future Improvements
- Deploy with Flask/Streamlit
- Try deep learning (LSTM/Transformer)
