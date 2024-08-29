# Data Privacy Enhancement
This project aims to classify text data as either sensitive or non-sensitive using Natural Language Processing (NLP) and Machine Learning techniques. It also identifies named entities within the text to enhance data privacy.

## Table of Contents
- Installation
- Usage
- Dataset
- Model
- Evaluation
- Contributing

## Installation
1. Clone the repository:
`git clone https://github.com/yourusername/data-privacy-enhancement.git`
- cd data-privacy-enhancement

2. Create a virtual environment:
- python -m venv venv
- source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
- pip install -r requirements.txt

4. Download the spaCy model:
- python -m spacy download en_core_web_trf

## Usage
1. Run the script:
- python data_privacy_enhancement.py

- Output: The script will output the classification report, accuracy score, named entities found, and the classification result for a new example text.

## Dataset
- The dataset consists of text samples labeled as either sensitive or non-sensitive. Sensitive data includes personal information such as credit card numbers, email addresses, and social security numbers.

## Model
- The project uses a Random Forest classifier with TF-IDF vectorization for text classification. Named Entity Recognition (NER) is performed using the en_core_web_trf model from spaCy.

## Evaluation
- The model is evaluated using precision, recall, F1-score, and accuracy. The current model achieves an accuracy score of 1.0 on the test set.

## Contributing
- Contributions are welcome! Please fork the repository and submit a pull request with your changes.
