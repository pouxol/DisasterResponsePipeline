import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('CatMess', con=engine)
    X = df['message']
    Y = df.drop(['message', 'original', 'id', 'genre'], axis=1)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words('english')]
        
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(
            KNeighborsClassifier(n_neighbors=8, weights='uniform')))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(data=Y_pred, columns=category_names)
    
    print(classification_report(Y_test, Y_pred, target_names=category_names))

    for col in category_names:
        print('Classification Report for '+ col)
        print(classification_report(Y_test[col], Y_pred[col]))
        print('----------------------------------------------------\n')
        
    
def save_model(model, model_filepath):
    CatMessClf = pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()