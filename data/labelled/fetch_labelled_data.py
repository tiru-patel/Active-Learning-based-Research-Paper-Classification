import pandas as pd
import json

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
        df_filtered.to_csv('labelled_data.csv')
    return df_filtered

if __name__== "__main__":
    main_df = decode_json(save_csv=True)
    labelled_data = get_CS_data(main_df, save_csv=True)
