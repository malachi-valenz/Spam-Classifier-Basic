# Spam Email Classifier

The Spam Email Classifier uses machine learning to categorize spam (junk/malicious) and ham (legitimate) content.
## Installation
Python 3.8+ Required

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pandas and scikit-learn. 

```bash
pip install pandas scikit-learn
```

## Usage

```bash
python spamemailclassifier.py
```

**Must have a file named spam.csv within the same folder for the data to be read**

The dataset used was the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

## Sample Output

```bash 
              precision    recall  f1-score   support

         ham       0.99      0.99      0.99       965
        spam       0.92      0.92      0.92       150

    accuracy                           0.98      1115
   macro avg       0.95      0.95      0.95      1115
weighted avg       0.98      0.98      0.98      1115

```

Enjoy!