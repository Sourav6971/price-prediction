import json
import pickle
__data_columns=None
__locations=None
__model=None

def get_location_names():
    pass



def load_saved_artefacts():
    print('loading artifacts.....start')
    global __data_columns
    global __locations                  
    with open('columns.json','r')as f:
        __data_columns=json.load(f)['data_columns'];   
        __locations=__data_columns[3:]
    with open('banglore_home_prices_model','rb')as f:
        __model= pickle.load(f);
    print('artefacts.....loaded')

    
if __name__=='__main__':
    print(get_location_names)