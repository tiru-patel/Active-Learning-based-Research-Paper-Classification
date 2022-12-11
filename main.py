import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_text
from features.vectorizer import Vectorizer
from utils.constants import CATEGORIES, TITLE, ABSTRACT, TEXT_FEATURE
from models.supervised_model import SupervisedModel
import warnings
import os
warnings.filterwarnings("ignore") 

df = pd.read_csv(os.getcwd() + "/data/labelled/" + "cs_data.csv")
df = df[[TITLE, ABSTRACT, CATEGORIES]]

mapping = {}
unique_cats = df[CATEGORIES].unique()
for index, cat in enumerate(unique_cats):
    mapping[cat] = index

df[CATEGORIES] = [mapping[cat] for cat in df[CATEGORIES]]

df[ABSTRACT] = preprocess_text(df, ABSTRACT)
df[TITLE] = preprocess_text(df, TITLE)


df[TEXT_FEATURE] = df[[TITLE, ABSTRACT]].agg(' '.join, axis=1)
df = df[[TEXT_FEATURE, CATEGORIES]]

vectorizer = Vectorizer(data=df)
cv = vectorizer.get_count_vectorizer()
tfidf = vectorizer.get_tfidf_vectorizer()

supervisedModel = SupervisedModel(cv, df[CATEGORIES])
supervisedModel.getBestModel()
