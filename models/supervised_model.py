from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

class SupervisedModel:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
        self.clf = None
        self.accuracy = None

    def getBestModel(self):
        model_param = [
            {'clf': [LogisticRegression()], 
            'clf__penalty': [ 'l2'],
            'clf__C': [1,0.5,10]},
                    
            {'clf': [KNeighborsClassifier()], 
            'clf__n_neighbors': [5, 10]},

            {'clf': [RandomForestClassifier()], 
            'classifier__n_estimators': [10, 50, 100, 250]},

            {'clf': [GaussianNB()], 
            'clf__var_smoothing': [0, -3]},

            {'clf': [SVC()], 
            'clf__classifier__C': [10**-2, 10**-1, 10**0, 10**1, 10**2]}
        ]
        modelSelector = ModelSelector(model_param, self.x_train, self.y_train)
        print(modelSelector.score_summary())

        

# if __name__ == '__main__':
#     iris = datasets.load_iris()
#     supervisedModel = SupervisedModel(iris.data, iris.target)
#     supervisedModel.getBestModel()

