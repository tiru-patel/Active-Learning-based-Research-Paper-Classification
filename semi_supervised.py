import numpy as np
import pandas as pd

from preprocess import preprocess_text
from features.vectorizer import Vectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer



class Semisupervised():

    def __init__(self, labelled_data, unlabelled_data):
        df = labelled_data[['title', 'abstract','categories']]
        df_unlab = unlabelled_data[['title', 'abstract']]

        df['title'] = preprocess_text(df, 'title')
        df['abstract'] = preprocess_text(df, 'abstract')

        df_unlab['title'] = preprocess_text(df_unlab, 'title')
        df_unlab['abstract'] = preprocess_text(df_unlab, 'abstract')

        unlab_x = df_unlab[['title', 'abstract']].agg(' '.join, axis=1)

        x = df[['title', 'abstract']].agg(' '.join, axis=1)
        y = df["categories"]

        train_x,test_x,train_y,test_y = tts(x,y,test_size=0.2,stratify=y)

        tfidf_vect = TfidfVectorizer(stop_words = 'english')
        
        tfidf_train = tfidf_vect.fit_transform(train_x)
        self.tfidf_test = tfidf_vect.transform(test_x)
        
        le = LabelEncoder()
        self.encoded_y = le.fit_transform(train_y)
        self.test_encoded_y = le.fit_transform(test_y)

        self.X_train_text = np.concatenate((np.array(train_x), np.array(unlab_x)))

        nolabel = [-1 for _ in range(len(unlab_x))]
        self.y_train_final = np.concatenate((self.encoded_y, nolabel))

        tfidf_semi_sup = tfidf_vect.transform(self.X_train_text)
        self.X_train_final = pd.DataFrame(tfidf_semi_sup.A, columns=tfidf_vect.get_feature_names_out())

    def train_label_propagation(self):
        # define model
        model = LabelPropagation()

        # fit model on training dataset
        model.fit(self.X_train_final, self.y_train_final)

        return model
    
    def predict_label_propagation(self, model):
        pred = model.predict(self.tfidf_test)

        return pred

    def train_self_training_classifier(self, base_model = LogisticRegression(solver='lbfgs', max_iter=1000)):
        # Specify Self-Training model parameters
        self_training_model = SelfTrainingClassifier(base_estimator=base_model)

        # Fit the model
        clf_self_training = self_training_model.fit(self.X_train_final, self.y_train_final)

        return clf_self_training

    def predict_self_training_classifier(self, model):
        pred = model.predict(self.tfidf_test)

        return pred









