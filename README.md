# Twitter Sentiment Analysis

> **A powerful deep learning solution for Twitter sentiment classification using Bidirectional LSTM with an interactive web deployment interface.**

## 🎯 Project Overview

This project provides an end-to-end natural language processing solution for **Twitter sentiment analysis** using advanced deep learning techniques. The system achieves **92% accuracy** in sentiment classification using Bidirectional LSTM architecture with comprehensive text preprocessing, all deployed through an intuitive web interface.

### 🔬 Key Features

- **Advanced NLP Model**: Bidirectional LSTM with embedding layers
- **High Performance**: **92% accuracy** in sentiment classification
- **Smart Text Processing**: Comprehensive tweet cleaning and preprocessing
- **Interactive Web Interface**: Flask-based deployment with real-time predictions
- **Multi-Class Classification**: Positive, Negative, and Neutral sentiment detection
- **Production Ready**: Complete deployment solution with optimized models

## 📊 Dataset Overview

### Twitter Sentiment Dataset
- **Training Data**: 74,682 labeled tweets
- **Validation Data**: 1,000 labeled tweets  
- **Classes**: Positive, Negative, Neutral
- **Format**: CSV files with text and sentiment labels
- **Preprocessing**: Advanced text cleaning and stopword removal


## 📁 Project Structure

```
twitter_sentiment_project/
│
├── 📁 Notebook/                     # Model Development
│   ├── Twitter_Sentiment_Model.ipynb
│   └── Twitter_Sentiment_Model.py
│
├── 📁 dataset/                      # Training Data
│   ├── twitter_training.csv
│   └── twitter_validation.csv
│
├── 📁 deployment/                   # Web Application
│   ├── app.py                      # Flask application server
│   ├── 📁 static/                  # Frontend assets
│   │   ├── 📁 css/
│   │   │   └── styles.css
│   │   └── 📁 js/
│   │       └── script.js
│   ├── 📁 templates/               # HTML templates
│   │   └── index.html
│   └── 📁 models/                  # Trained Models
│       ├── sentiment_lstm_model_acc_0.92.h5
│       ├── tokenizer_acc_0.92.pkl
│       └── label_encoder_acc_0.92.pkl
│
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # Project documentation
└── 📄 LICENSE                      # MIT License
```

## 🎯 Model Performance

### Bidirectional LSTM Results
- **Test Accuracy**: 92%
- **Architecture**: 
  - Embedding Layer (128 dimensions)
  - 2x Bidirectional LSTM layers (32, 16 units)
  - Dense layers with dropout (64, 32 units)
  - Softmax output (3 classes)
- **Training**: 10 epochs with early stopping
- **Optimization**: Adam optimizer with categorical crossentropy loss

### Classification Report
```
              precision    recall  f1-score   support
    Negative       0.93      0.97      0.95       266
     Neutral       0.96      0.82      0.88       285
    Positive       0.87      0.96      0.91       277
    accuracy                           0.92       828
```

## 🔧 Text Preprocessing Pipeline

### Advanced Cleaning Features
- **URL Removal**: Eliminates links and web addresses
- **Mention Cleaning**: Removes @username mentions
- **HTML Tag Removal**: Strips HTML formatting
- **Special Character Filtering**: Keeps only alphabetic characters
- **Smart Stopword Removal**: Preserves negation words (not, never, no)
- **Tokenization**: Converts text to sequences for LSTM processing

## 🤝 Contributing

Feel free to suggest improvements, fix bugs, or add new features.  
Thanks for your support.

## 📞 Contact

**Ahmed Maher Abd Rabbo**
- 💼 [LinkedIn](https://www.linkedin.com/in/ahmed-maherr/)
- 📊 [Kaggle](https://kaggle.com/ahmedmaherabdrabbo)
- 📧 Email: ahmedbnmaher1@gmail.com
- 💻 [GitHub](https://github.com/AhmedMaherAbdRabbo)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.