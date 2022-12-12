import pandas as pd
from utils.constants import CATEGORIES
from utils.config import OVERSAMPLER_K
from sklearn.neighbors import KNeighborsClassifier
import warnings
from tqdm import tqdm
import pickle
warnings.filterwarnings("ignore")

class OverSampler:

    def __init__(self, data:pd.DataFrame, label: int):
        """
        Initialize the data variables, prepare feature
        set and target variable, and filter out the
         members of the given minority class
        :param data: Dataframe object pandas
        :param label: Minority class label
        """
        self.cv = data
        self.y = self.cv[CATEGORIES]
        self.X = self.cv.drop(columns=[CATEGORIES])
        self.label = label
        self.minority_labels = self.cv[self.cv['categories'] == self.label]
        self.minority_labels = self.minority_labels.reset_index(drop=True)

    def get_knn_matrix(self) -> pd.DataFrame:
        """
        Generate the impurity ratio of all members of minority
        class for multiple values of nearest neighbor K.
        Consolidates the result as a single dataframe
        and returns.
        :return: Dataframe object pandas
        """

        result = pd.DataFrame()
        for k in OVERSAMPLER_K:
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(self.X, self.y)
            x = None
            ir = []
            for index in tqdm(range(len(self.minority_labels))):
                record = self.minority_labels.loc[index, :].reset_index().T
                record = record[record.columns[:-1]]
                record = record.drop('index')
                x = record
                n = neigh.kneighbors(x)
                labels = self.cv.loc[n[1].tolist()[0]].reset_index(drop=True)
                ir.append(
                    len(labels[labels[CATEGORIES] == self.label]) /
                    len(labels)
                )
            result["k" + str(k)] = ir
        return result

# if __name__ == '__main__':
#
#     with open("features/count_vectorizer.pkl", 'rb') as f:
#         data = pickle.load(f)
#
#     oversampler = OverSampler(data=data)
#     result = oversampler.get_knn_matrix()
#
#     result.to_pickle("sampling_final_result.pkl")
#     print(result.mean(axis=0))



