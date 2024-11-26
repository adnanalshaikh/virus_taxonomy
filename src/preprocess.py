import os
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

HOME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DataPreProcess:
    def __init__(self, dna_filename, node, config):
        self.config = config
        dna_filename_path = os.path.join(HOME_DIR, 'data', dna_filename)
        df = pd.read_csv(dna_filename_path)
         
        species_length_thresh = self.config.get('species_length_thresh', 100)
        df = self.remove_short_sequences(df, species_length_thresh)
        
        num_species_thresh = self.config.get('num_species_thresh', 5)
        df = self.remove_targets_below_threshold(df, 'family', num_species_thresh)
        
        classifier = self.config.get('classifier', 'family')
        df = self.classifier_data (df, classifier, node)
        self.classifier_name = node
        
        self.seq_type = self.config.get('seq_type', 'seq')
        if self.seq_type != 'seq':
            df = df.rename(columns={self.seq_type: 'seq'})
        
        self.terget_to_index = {taxa: i for i, taxa in enumerate(df['target'].unique())}
        print("targets_dict: ", self.terget_to_index)
        
        df['target'] = df['target'].map(self.terget_to_index)
        df['id'] = df.index
        
        df_taxo = df[['id', 'target', 'order', 'family']].copy()
        df = df[['id', 'seq', 'target', 'order', 'family']]

        #print(df[['id', 'target']].assign(seq_length=df['seq'].apply(len)))

        self.nucleic_acids = config.get('nucleic_acids', 'DNA')
        
        if self.nucleic_acids in ['DNA', 'RNA']:
            df = self.handle_bad_sequence(df, sequence_type=self.nucleic_acids)
            df = self.add_reverse_complement(df, sequence_type=self.nucleic_acids)
            self.df_subseq = self.encode_scale_split(df, sequence_type=self.nucleic_acids)
        else:
            raise ValueError(f"Unrecognized nucleic_acids type: {self.nucleic_acids}")

    def classifier_data (self, df, classifier, node):
        self.seq_type = self.config.get('seq_type', 'seq')
        
        if classifier == 'order':
            df = df[df['class'] == node][[self.seq_type, 'order', 'family']]
            df['target'] = df['order']
            #df = df.rename(columns={'order': 'target'})
        elif classifier == 'family':
            df = df[df['order'] == node][[self.seq_type, 'order', 'family']]
            #df = df.rename(columns={'family': 'target'})
            df['target'] = df['family']
        else:
            raise ValueError("Unrecognized nucleic_acids type")
        return df
                          
    def classifier_data_1 (self, df, classifier, node):
        self.seq_type = self.config.get('seq_type', 'seq')
        
        if classifier == 'order':
            df = df[df['class'] == node][[self.seq_type, 'order']]
            df = df.rename(columns={'order': 'target'})
        elif classifier == 'family':
            df = df[df['order'] == node][[self.seq_type, 'family']]
            df = df.rename(columns={'family': 'target'})
        else:
            raise ValueError("Unrecognized nucleic_acids type")
        return df
        
    def remove_targets_below_threshold(self, df, target_column, threshold):
        target_counts = df[target_column].value_counts()
        df_filtered = df[df[target_column].isin(target_counts[target_counts >= threshold].index)]
        return df_filtered.reset_index(drop=True)

    def remove_short_sequences(self, df, threshold):
        df = df[~df['seq'].isna()]
        df = df[df['seq'].apply(len) >= threshold].reset_index(drop=True)
        return df
    
    def handle_bad_sequence(self, df, sequence_type='DNA'):
        print(f'Handling bad {sequence_type}...')
        
        def clean_sequence(seq):
            counts = Counter(seq)
            if sequence_type == 'DNA':
                valid_bases = "ACGT"
            elif sequence_type == 'RNA':
                valid_bases = "ACGU"
            else:
                raise ValueError(f"Invalid sequence type: {sequence_type}. Expected 'DNA' or 'RNA'.")
            
            # Find the most frequent valid base
            most_freq = max({char: counts[char] for char in counts if char in valid_bases}, key=counts.get, default='A')
            
            # Replace invalid bases with the most frequent valid base
            return ''.join(most_freq if char not in valid_bases else char for char in seq)
        
        df['seq'] = df['seq'].apply(clean_sequence)
        return df

    def one_hot_encode(self, seq, sequence_type='DNA'):
        if sequence_type == 'DNA':
            mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        elif sequence_type == 'RNA':
            mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}
        else:
            raise ValueError(f"Invalid sequence type: {sequence_type}. Expected 'DNA' or 'RNA'.")
        
        return np.array([mapping.get(nuc, [0, 0, 0, 0]) for nuc in seq])  # Handle invalid characters
    
    def reverse_complement(self, seq, sequence_type='DNA'):
        if sequence_type == 'DNA':
            complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        elif sequence_type == 'RNA':
            complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
        else:
            raise ValueError(f"Invalid sequence type: {sequence_type}. Expected 'DNA' or 'RNA'.")
        
        return ''.join(complement.get(base, base) for base in reversed(seq))
    
    def add_reverse_complement(self, df, sequence_type='DNA'):
        print(f"Adding reverse complement for {sequence_type}...")
        df_complement = df.copy()
        df_complement['seq'] = df_complement['seq'].apply(lambda seq: self.reverse_complement(seq, sequence_type))
        
        # Assign the 'is_complement_of' column to store the original ID
        df_complement['is_complement_of'] = df_complement['id']
        
        # Create new unique IDs for the reverse complement sequences
        df_complement['id'] = df_complement['id'].max() + df_complement.index + 1
        
        # Mark original sequences with 'None' for 'is_complement_of'
        df['is_complement_of'] = None
        
        # Concatenate the original and complement sequences
        return pd.concat([df, df_complement], ignore_index=True)


    def encode_scale_split(self, df, sequence_type='DNA'):
        print(f'Encoding, scaling, and splitting {sequence_type} sequences...')
        
        subseq_len = self.config.get('subseq_len', None)
        stride = self.config.get('stride', subseq_len)  # For DNA
        subseq_num = self.config.get('subseq_num', subseq_len)  # For DNA
        
        new_rows = []
        
        for _, row in df.iterrows():
            # One-hot encode based on sequence type
            encoded_seq = self.one_hot_encode(row['seq'], sequence_type=sequence_type)
            subsequences = self.split_sequence_equal_subsequences(encoded_seq, subseq_len, subseq_num)          
            # Add subsequences to new rows
            for subseq in subsequences:
                new_rows.append({'id': row['id'], 'seq': subseq, 'target': row['target'], 
                                 'is_complement_of': row['is_complement_of'], 
                                 'order': row['order'], 'family': row['family']})
        
        return pd.DataFrame(new_rows)
   
    
    def split_sequence_equal_subsequences(self, sequence, subseq_len, num_subsequences):
        """
        Split the sequence into `num_subsequences` by adjusting the stride dynamically.
        The subseq_len specifies the length of each subsequence. Ensure no subsequence is shorter than subseq_len.
        """
        seq_len = len(sequence)
        
        # If the sequence is shorter than the subseq_len, return an empty list or handle as needed
        if seq_len < subseq_len:
            return []  # Return empty list as we can't split into valid subsequences
        
        # Calculate the stride dynamically based on the length of the sequence
        stride = max(1, (seq_len - subseq_len) // (num_subsequences - 1))
        
        subsequences = []
        for i in range(0, seq_len - subseq_len + 1, stride):
            subsequences.append(sequence[i:i + subseq_len])
            if len(subsequences) == num_subsequences:  # Stop when the required number of subsequences is reached
                break
        
        return subsequences
       
    def ensure_no_leakage_train_val(self, df_train_test):
        """
        Ensure no data leakage in train/validation split:
        1) All subsequences of a sequence are in either train or val set.
        2) None of the subsequences of validation sequences are in the train set.
        3) None of the subsequences of reverse complements of validation sequences are in the train set.
        """
        # Step 1: Get the unique sequence IDs in df_train_test
        unique_ids = df_train_test['id'].unique()
        
        # Step 2: Perform the train/test split using these unique sequence IDs
        train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, stratify=df_train_test['target'], random_state=42)
    
        # Step 3: Create the validation and training sets
        df_val = df_train_test[df_train_test['id'].isin(val_ids)]
        df_train = df_train_test[df_train_test['id'].isin(train_ids)]
    
        # Step 4: Ensure that none of the reverse complements of validation sequences are in the train set
        val_complement_ids = df_train_test[df_train_test['is_complement_of'].isin(val_ids)]['id'].unique()
    
        # Remove all reverse complement subsequences of validation sequences from the training set
        df_train = df_train[~df_train['is_complement_of'].isin(val_ids) & ~df_train['id'].isin(val_complement_ids)]
    
        return df_train, df_val
    
    def get_KFold_Samples(self, n_splits=5):
        df = self.df_subseq
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        original_df = df[df['is_complement_of'].isna()]
        unique_ids = original_df['id'].unique()
        id_to_target = original_df[['id', 'target']].drop_duplicates().set_index('id')
        
        for train_idx, test_idx in skf.split(unique_ids, id_to_target.loc[unique_ids, 'target']):
            train_ids = unique_ids[train_idx]
            test_ids = unique_ids[test_idx]
            df_train = df[(df['id'].isin(train_ids)) | (df['is_complement_of'].isin(train_ids))]
            df_test = df[(df['id'].isin(test_ids)) & (df['is_complement_of'].isna())]
            df_train = df_train[~df_train['id'].isin(df_test['is_complement_of'].dropna())]
            df_test = df_test[~df_test['is_complement_of'].isin(df_train['id'].dropna())]
            
            # Check for overlap between training and testing IDs
            assert not any(df_train['id'].isin(df_test['id'])), "Training IDs overlap with Testing IDs!"
            assert not any(df_train['id'].isin(df_test['is_complement_of'].dropna())), "Training includes reverse complements from Testing!"
            assert not any(df_test['id'].isin(df_train['is_complement_of'].dropna())), "Testing includes reverse complements from Training!"

            # Check if any reverse complements overlap between training and testing
            overlap_complements = set(df_train['id']).intersection(set(df_test['is_complement_of'].dropna()))
            print(f"Overlapping Reverse Complements: {overlap_complements}")

            yield df_train, df_test