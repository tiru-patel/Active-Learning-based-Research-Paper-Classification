import urllib, urllib.request
import xmltodict
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import time

categories = {"Artificial Intelligence": "all:machine+AND+all:learning",
              "Hardware Architecture": "all:hardware+AND+all:architecture",
              "DSA": "all:data+AND+all:structures+AND+all:and+AND+all:algorithm",
              "Information Retrieval": "all:information+AND+all:retrieval", 
              "Human Computer Interaction": "all:human+AND+all:computer+AND+all:interaction",
              "Cryptography": "all:cryptography+AND+all:and+AND+all:security",
              "Networking": "all:networking+AND+all:and+AND+all:internet+AND+all:architecture" 
             }

labelled_data = pd.DataFrame()
for c,q in categories.items():
    print(c)
    url = 'http://export.arxiv.org/api/query?search_query={}&start=10&max_results=4000'.format(q)
    data = urllib.request.urlopen(url)
    doc = data.read().decode('utf-8')
    data_dict = xmltodict.parse(doc)
    count=0
    while count<len(data_dict['feed']['entry']):
        new_row = {'Title': data_dict['feed']['entry'][count]["title"],
                   'Abstract': data_dict['feed']['entry'][count]["summary"], 
                   'Domain': c}
        #append row to the dataframe
        labelled_data = labelled_data.append(new_row, ignore_index=True)
        count+=1  
        time.sleep(60)

# Sample the data
labelled_data = labelled_data.sample(frac=1).reset_index()
# Save the dataframe to CSV
labelled_data.to_csv("API_unlabelled.csv")