import pandas as pd
import json

from sklearn.model_selection import train_test_split

# Update filename
def decode_json(filename, save_csv=False):
    with open(filename) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    if save_csv==True:
        df_final.to_csv("df_final.csv")
    return df_final

def get_CS_data(df_final, save_csv=False):

    searchfor = ['cs.AI', 'cs.AR', 'cs.HC', 'cs.DS', 'cs.IR','cs.CR', 'cs.NI']

    df_filtered = df_final.loc[df_final['categories'].isin(searchfor)]
    df_filtered.reset_index(inplace=True)
    df_filtered = df_filtered.drop(columns=['index'], axis=1)
    if save_csv==True:
        df_filtered.to_csv('cs_data.csv')
    return df_filtered

def get_active_learning_data(data, dataset_ratio = 0.2, save_csv_labelled = False, save_csv_unlabelled=False):
    df_labelled, df_unlabelled = train_test_split(data, train_size = dataset_ratio, stratify=data['categories'], random_state=42)

    if save_csv_labelled == True:
        df_labelled.to_csv('labelled_data.csv')

    if save_csv_unlabelled == True:
        df_unlabelled.to_csv('../unlabelled_data/unlabelled_data_json.csv')

    return df_labelled, df_unlabelled

if __name__== "__main__":
    main_df = decode_json('arxiv-metadata-oai-snapshot.json',save_csv=True)
    cs_data = get_CS_data(main_df, save_csv=True)

    labelled_data, unlabelled_data = get_active_learning_data(cs_data, 0.2, True, True)




