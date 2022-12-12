import warnings
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split as tts
import pandas as pd
from numpy import loadtxt
from sklearn.feature_extraction.text import TfidfVectorizer


def batch_creator(df, batch_size):
    return (df[pos: pos + batch_size] for pos in range(0, len(df), batch_size))


def make_prediction(data, tfidf_vect, lg):
    batch_vect = tfidf_vect.transform(data.to_numpy()).toarray()
    #batch_vect_spca = transformer.transform(batch_vect)
    batch_prediction = lg.predict(batch_vect)
    prob_prediction = lg.predict_proba(batch_vect)
    
    return batch_prediction, prob_prediction

def active_learning_model(traindata, trainlabels, lg):
    x = traindata
    #print(x.shape)
    y = trainlabels
    
    active_lg = lg.fit(x, y)

    active_train_preds = lg.predict(x)
    active_preds = lg.predict(test_x)
    
    train_active_acc = accuracy_score(y, active_train_preds)
    active_acc = accuracy_score(test_y, active_preds)
    print()
    print(classification_report(test_y, active_preds))
    return active_lg, active_acc, train_active_acc

def active_learning(batch_size, entropy_cutoff, probability_cutoff, train_x, train_y, tfidf_vect, unlabelled_data, lg):
    batch_num = 0
    batch_size=batch_size
    cutoff=entropy_cutoff
    cutoff_prob = probability_cutoff
    train_data = train_x 
    train_labels = train_y

    TrainAcc = []
    TestAcc = []
    samples_added = []

    for batch in batch_creator(unlabelled_data, batch_size):
        sampled_pool = pd.DataFrame()
        predictions, probabilities = make_prediction(batch["text"], tfidf_vect, lg)
        try:
            for index, data in batch.iterrows():
                if(unlab_entropy[index] >= cutoff and np.max(probabilities[index - batch_num * batch_size])<=cutoff_prob):
                    sampled_pool = sampled_pool.append({
                        'text': data['text'],
                        'categories': predictions[index - batch_num * batch_size]
                    }, ignore_index=True)     


            new_data = tfidf_vect.transform(sampled_pool["text"])
            new_data = pd.DataFrame(new_data.A, columns=tfidf_vect.get_feature_names_out())
            train_data = np.vstack((train_data, new_data))

            new_labels = sampled_pool["categories"]
            train_labels = train_labels.append(new_labels)

            active_model, active_score, train_active_score = active_learning_model(train_data, train_labels, lg)
            TrainAcc.append(train_active_score)
            TestAcc.append(active_score)
            samples_added.append(train_data.shape[0] - new_data.shape[0])
            print("Testing score : ", active_score)
            print("Training score : ", train_active_score)
            print(train_data.shape)
            print("***********************")
            print()
            batch_num += 1
        except Exception as e: 
            continue

    return TrainAcc, TestAcc, samples_added, active_model