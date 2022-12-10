import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_text
import warnings
warnings.filterwarnings("ignore") 

df = pd.read_csv('labelled_data.csv')
df= df[['title', 'abstract', 'categories']]

df["abstract"] = preprocess_text(df, "abstract")
df["title"] = preprocess_text(df, "title")

print(df.head())