from utils.constants import TEXT_FEATURE, CATEGORIES, COUNT_VECT, TFIDF_VECT
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

class Vectorizer:

    def __init__(self,
                 data:pd.DataFrame = None):
        """
        Constructor for defining input data
        :param data: Dataframe object pandas
        """
        self.data = data
        self.cv_model = None
        self.cv_data = None

    def store_vectors(self,
                      filename: str,
                      vectors_df: pd.DataFrame
    ):
        """
        Save vectors into cwd
        :param filename: string name assigned to saved file
        :param vectors_df: Dataframe object pandas
        :return: None. Saves te file
        """
        vectors_df[CATEGORIES] = self.data[CATEGORIES]
        vectors_df.to_pickle(filename + '.pkl')
        # with open(filename+'.pkl', 'wb') as handle:
        #     pickle.dump(vectors_df, handle)

    def get_vector_df(self, vectors, vectorizer, name:str):

        vectors_df = pd.DataFrame(vectors.A)
        vectors_df.columns = vectorizer.vocabulary_.keys()
        if name == COUNT_VECT:
            self.cv_data = vectors_df
        self.store_vectors(filename=name,vectors_df=vectors_df)
        return vectors_df

    def get_count_vectorizer(self, feature: str = TEXT_FEATURE):
        """
        To generate scikit-learn CountVectorizer
        features for the input data
        :param feature: feature name to be vectorized
        :return: DataFrame object of features
        """
        logging.info("Preparing CountVectorizer features.....")

        count_vectorizer = CountVectorizer(stop_words='english')
        vectors = count_vectorizer.fit_transform(self.data[feature])

        logging.info("CountVectorizer Features Generated!")
        self.cv_model = count_vectorizer
        return self.get_vector_df(vectors=vectors,
                                  vectorizer=count_vectorizer,
                                  name=COUNT_VECT)

    def get_tfidf_vectorizer(self, feature: str = TEXT_FEATURE):
        """
        To generate scikit-learn TfidfVectorizer
        features for the input data
        :param feature: feature name to be vectorized
        :return: DataFrame object of features
        """
        logging.info("Preparing TFIDF features.....")

        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        vectors = tfidf_vectorizer.fit_transform(self.data[feature])

        logging.info("TFIDF Features Generated!")
        return self.get_vector_df(vectors=vectors,
                                  vectorizer=tfidf_vectorizer,
                                  name=TFIDF_VECT)
