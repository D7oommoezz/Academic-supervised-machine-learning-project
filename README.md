# Glass Classification - Academic Machine Learning Project

This repository contains a cleaned and documented version of an academic supervised machine learning project for classifying glass types based on numerical chemical/physical features.

The original project was built during a data mining / machine learning course using Python and scikit-learn. This version reorganizes the code into a clearer GitHub-ready structure.

## Project Objective

The goal is to train classification models that predict the `Type of glass` from input features such as:

- Refractive Index (RI)
- Sodium (Na)
- Magnesium (Mg)
- Aluminum (Al)
- Silicon (Si)
- Potassium (K)
- Calcium (Ca)
- Barium (Ba)
- Iron (Fe)

## Dataset

The dataset is stored in:

```text
data/glass.csv
```

It contains:

- 214 records
- 9 input features
- 1 target label: `Type of glass`

Class distribution:

| Glass Type | Count |
|---:|---:|
| 1 | 70 |
| 2 | 76 |
| 3 | 17 |
| 5 | 13 |
| 6 | 9 |
| 7 | 29 |

## Methods Used

The cleaned implementation includes:

- Data loading with pandas
- Feature/target separation
- Train/test split with stratification
- Feature scaling where needed
- Model training with scikit-learn
- Accuracy score
- Classification report
- Confusion matrix
- Model comparison

## Models Compared

The script compares several supervised classification models:

- Gaussian Naive Bayes
- Logistic Regression
- Support Vector Machine (RBF Kernel)
- Random Forest Classifier

## Current Results

Using an 80/20 stratified train-test split with `random_state=42`:

| Model | Accuracy |
|---|---:|
| Gaussian Naive Bayes | 51.16% |
| Logistic Regression | 72.09% |
| SVM (RBF Kernel) | 72.09% |
| Random Forest | 81.40% |

The best-performing model in this run is Random Forest.

> Note: This is an academic project and not a production machine learning system. The purpose is to demonstrate understanding of supervised learning, classification, preprocessing, training, and evaluation.

## Project Structure

```text
glass-classification-ml/
│
├── data/
│   └── glass.csv
│
├── src/
│   └── train.py
│
├── legacy/
│   └── Glass_Classification_original.py
│
├── results/
│   ├── model_results.csv
│   └── confusion_matrix_random_forest.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

The dataset file used by the project is `data/glass.csv`. The main script already points to this file directly.


1. Clone the repository:

```bash
git clone <repository-url>
cd glass-classification-ml
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate it:

Windows:

```bash
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the training script:

```bash
python src/train.py
```

## Skills Demonstrated

- Python
- pandas
- NumPy
- scikit-learn
- Supervised learning
- Classification
- Train/test split
- Feature scaling
- Model evaluation
- Confusion matrix analysis
- Basic machine learning workflow organization

## Future Improvements

Possible improvements include:

- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Better handling of class imbalance
- Additional visualizations
- Saving trained models using joblib
- Building a small inference script or simple web demo
