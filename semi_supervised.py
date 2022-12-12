import numpy as np
import pandas as pd

from preprocess import preprocess_text
from word2vec import Word2Vec
from features.vectorizer import Vectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import accuracy_score


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

        train_x,test_x,train_y,test_y = tts(x,y,train_size=0.6,stratify=y)
        self.test_labels = test_y

        tfidf_vect = TfidfVectorizer(stop_words = 'english', max_features=1000)
        
        tfidf_train = tfidf_vect.fit_transform(train_x)
        self.tfidf_test = tfidf_vect.transform(test_x)

        sm2 = BorderlineSMOTE(random_state=42)
        tfidf_train, train_y = sm2.fit_resample(tfidf_train, train_y)

        tfidf_unlab = tfidf_vect.transform(unlab_x)

        tfidf_train_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names_out())
        tfidf_unlab_df = pd.DataFrame(tfidf_unlab.A, columns=tfidf_vect.get_feature_names_out())

        le = LabelEncoder()
        self.encoded_y = le.fit_transform(train_y)
        self.test_encoded_y = le.fit_transform(test_y)

        self.X_train_text = tfidf_train_df.append(tfidf_unlab_df, ignore_index=True)

        nolabel = [-1 for _ in range(len(unlab_x))]
        self.y_train_final = np.concatenate((self.encoded_y, nolabel))

        self.X_train_final = pd.DataFrame(self.X_train_text, columns=tfidf_vect.get_feature_names_out())

    def train_label_propagation(self):
        # define model
        model = LabelPropagation()

        # fit model on training dataset
        model.fit(self.X_train_final, self.y_train_final)

        return model
    
    def predict_label_propagation(self, model):
        pred = model.predict(self.tfidf_test)
        acc = accuracy_score(self.test_encoded_y, pred)

        return pred, acc

    def train_self_training_classifier(self, base_model = LogisticRegression(solver='lbfgs', max_iter=1000)):
        # Specify Self-Training model parameters
        self_training_model = SelfTrainingClassifier(base_estimator=base_model)

        # Fit the model
        clf_self_training = self_training_model.fit(self.X_train_final, self.y_train_final)

        return clf_self_training

    def predict_self_training_classifier(self, model):
        pred = model.predict(self.tfidf_test)
        acc = accuracy_score(self.test_encoded_y, pred)

        return pred, acc









