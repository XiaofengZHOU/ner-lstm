import pickle

def get_data(file_name):
    file =  open(file_name,'rb')
    pickle_data = pickle.load(file)
    data = pickle_data['train_data']
    label = pickle_data['train_label']
    file.close()
    return data,label