import numpy as np
import pandas as pd
import gensim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Word2Vec:

    def __init__(self, data: pd.DataFrame = None, google_model_path = 'GoogleNews-vectors-negative300.bin'):
        
        # Initializing data and corpus
        self.data = data
        self.corpus = data['text'].values

        # Creating the count vectorizer
        self.vectorizer = CountVectorizer(stop_words='english')
        
        # Converting the text to numeric data
        X = self.vectorizer.fit_transform(self.corpus)
        
        # Saving the count vectorized data
        self.CountVectorizedData = pd.DataFrame(
            X.toarray(), 
            columns = self.vectorizer.get_feature_names()
        )

        # Adding new column for the target variable
        self.CountVectorizedData['Priority'] = self.data['categories']

        #Loading the word vectors from Google trained word2Vec model
        self.GoogleModel = gensim.models.KeyedVectors.load_word2vec_format(google_model_path, binary=True,)
        
        # Creating the list of words which are present in the Document term matrix
        self.WordsVocab = self.CountVectorizedData.columns[:-1]
                
    # Function for converting single line of text to vectorized form 
    def FunctionText2Vec(self, inpTextData):
        # Converting the text to numeric data
        X = self.vectorizer.transform(inpTextData)
        
        CountVecData = pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names())
        
        # Creating empty dataframe to hold sentences
        W2Vec_Data=pd.DataFrame()
        
        # Looping through each row for the data
        for i in range(CountVecData.shape[0]):
    
            # initiating a sentence with all zeros
            Sentence = np.zeros(300)
    
            # Looping thru each word in the sentence and if its present in 
            # the Word2Vec model then storing its vector
            for word in self.WordsVocab[CountVecData.iloc[i,:] >= 1]:
                #print(word)
                if word in self.GoogleModel.key_to_index.keys():    
                    Sentence=Sentence + self.GoogleModel[word]
            # Appending the sentence to the dataframe
            W2Vec_Data = W2Vec_Data.append(pd.DataFrame([Sentence]))
        
        return W2Vec_Data

    def convert_all_text_to_word2vec(self):
        self.W2Vec_Data = self.FunctionText2Vec(self.data['text'])
 
    def get_data_for_classification(self):
        # Adding the target variable
        self.W2Vec_Data.reset_index(inplace=True, drop=True)

        self.W2Vec_Data['Priority'] = self.CountVectorizedData['Priority']
        
        # Assigning to DataForML variable
        self.DataForML = self.W2Vec_Data

    def get_train_and_test_data(self):
        TargetVariable = self.DataForML.columns[-1]
        Predictors = self.DataForML.columns[:-1]
        
        X = self.DataForML[Predictors].values
        y = self.DataForML[TargetVariable].values

        PredictorScaler = StandardScaler()
        
        # Storing the fit object for later reference
        self.PredictorScalerFit = PredictorScaler.fit(X)
        
        # Generating the standardized values of X
        X = self.PredictorScalerFit.transform(X)
        
        # Split the data into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        return X_train, X_test, y_train, y_test

    def get_predictions_word2vec(self, model, input):
        X = self.FunctionText2Vec(input)
    
        X = self.PredictorScalerFit.transform(X)
        
        # Generating the prediction using Naive Bayes model and returning
        Prediction = model.predict(X)
        
        Result = pd.DataFrame(data = input, columns=['Text'])
        
        Result['Prediction']=Prediction
        
        return Result


    

