# don't forget hard coded 20 in ProcessData 
from measures import Measures
import os

from preprocess import  DataPreProcess
from cnn_model import CNNModel
HOME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def Martellivirales_model() :
    config = { 
        'modelid': 6,
        'batch_size': 10,
        'es_patience': 20,
        'learning_rate': 0.0002,
        'epochs': 800,
        'subseq_len': 1000,
        'subseq_num' : 20,
        'stride' : 1000,
        'k_folds': 5,
        'seq_type' : 'seq',
        'num_species_thresh' : 15,
        'species_length_thresh' : 1000,
        'nucleic_acids' : 'DNA',
        'classifier' : 'family',
        'news_measures_file': None, #'measures.csv',  # or None
        'kmer' : 4
        }

    dp =  DataPreProcess('Martellivirales.csv', 'Martellivirales', config)
    nmf = config.get ('news_measures_file', None)   
    measures = Measures(dp, filename=nmf)
                              

    model = CNNModel(measures, config)
    model.train_test_cv(dp)
    measures.write_measures()
    
    import pandas as pd
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping of columns
    pd.set_option('display.max_colwidth', None)  # Show full width of the column content
    print(measures)
    return measures

def Tymovirales_refseq_model() :
    config = { 
        'modelid': 6,
        'batch_size': 10,
        'es_patience': 15,
        'learning_rate': 0.000015,
        'epochs': 200,
        'subseq_len': 1000,
        'subseq_num' : 20,
        'stride' : 1000,
        'k_folds': 5,
        'seq_type' : 'seq',
        'num_species_thresh' : 15,
        'species_length_thresh' : 1000,
        'nucleic_acids' : 'DNA',
        'news_measures_file': None, #'measures.csv',  # or None
        }

    dp =  DataPreProcess('Tymovirales_refseq.csv', 'Tymovirales', config)
    
    nmf = config.get ('news_measures_file', None)   
    measures = Measures(dp, filename=nmf)
                              

    print("taxas : ", dp.terget_to_index)
    model = CNNModel(measures, config)
    model.train_test_cv(dp)
    measures.write_measures()
    
    import pandas as pd
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping of columns
    pd.set_option('display.max_colwidth', None)  # Show full width of the column content
    print(measures)
    return measures

def Alsuviricetes_refseq_heir_model() :
    config = { 
        'modelid': 6,
        'batch_size': 10,
        'es_patience': 30,
        'learning_rate': 0.0002, # not used 
        'epochs': 400,
        'subseq_len': 1000,
        'subseq_num' : 20,
        'stride' : 1000,
        'k_folds': 5,
        'seq_type' : 'seq',
        'num_species_thresh' : 15,
        'species_length_thresh' : 1000,
        'nucleic_acids' : 'DNA',
        'news_measures_file': None, #'measures.csv',  # or None
        'classifier' : 'order', 
        'kmer' : 4
        }

    dp =  DataPreProcess('Alsuviricetes_refseq.csv', 'Alsuviricetes', config)
    nmf = config.get ('news_measures_file', None)   
    measures = Measures(dp, filename=nmf)
                              

    print("taxas : ", dp.terget_to_index)
    model = CNNModel(measures, config)
    model.train_heirarchy_cv(dp)
    measures.write_measures()
    
    import pandas as pd
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping of columns
    pd.set_option('display.max_colwidth', None)  # Show full width of the column content
    print(measures)
    return measures

def Alsuviricetes_model() :
    config = { 
        'modelid': 6,
        'batch_size': 10,
        'es_patience': 30,
        'learning_rate': 0.0002, # not used
        'epochs': 200,
        'subseq_len': 1000,
        'subseq_num' : 20,
        'stride' : 1000,
        'k_folds': 5,
        'seq_type' : 'seq',
        'num_species_thresh' : 15,
        'species_length_thresh' : 1000,
        'nucleic_acids' : 'DNA',
        'news_measures_file': None, #'measures.csv',  # or None
        'classifier' : 'order', 
        'kmer' : 4
        }

    dp =  DataPreProcess('Alsuviricetes_refseq.csv', 'Alsuviricetes', config)
    nmf = config.get ('news_measures_file', None)   
    measures = Measures(dp, filename=nmf)
                              

    print("taxas : ", dp.terget_to_index)
    model = CNNModel(measures, config)
    model.train_test_cv(dp)
    measures.write_measures()
    
    import pandas as pd
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping of columns
    pd.set_option('display.max_colwidth', None)  # Show full width of the column content
    print(measures)
    return measures


#m = Martellivirales_model()
#m = Tymovirales_refseq_model()
#m = Alsuviricetes_model()
m = Alsuviricetes_refseq_heir_model()


   




