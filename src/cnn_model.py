# don't forget hard coded 20 in ProcessData 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, AveragePooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.utils import class_weight
import numpy as np
import os
import gc
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

HOME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CNNModel:
    def __init__(self, measures, config):
        self.config = config
        self.optimizer = Adam(config['learning_rate'])
        modelid = config['modelid']
        self.checkpoints_path = os.path.join(HOME_DIR, f"checkpoints/bestmodel_{modelid}.hdf5")
        self.measures = measures
        
    def build_model(self, input_shape, number_classes):
        modelid = self.config['modelid']
        self.optimizer = Adam(self.config['learning_rate'])
        
        if modelid == 1:
            return self.build_model_m1(input_shape, number_classes)
        elif modelid == 6:
            return self.build_model_m6(input_shape, number_classes)
        elif modelid == 7:
            return self.build_model_m7(input_shape, number_classes)

        else :
            return None

     
    def build_model_m6(self, input_shape, number_classes):
        
        print ("running model 6")

        model = Sequential()
        
        model.add(Conv1D(filters=64, kernel_size=3, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Conv1D(filters=64, kernel_size=3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling1D(pool_size=2))

        model.add(Conv1D(filters=128, kernel_size=3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(AveragePooling1D(pool_size=2))
        #model.add(Dropout(.5))
        
        model.add(Conv1D(filters=128, kernel_size=3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(AveragePooling1D(pool_size=2))
    
        model.add(LSTM(256, return_sequences=True)) 
        model.add(BatchNormalization())

        
        model.add(Flatten()) 
    
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.5))
        
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.5))
  
        #model.add(GlobalAveragePooling1D())          
        model.add(Dense(number_classes, activation='softmax'))  
        model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        return model
    

    def build_model_m7(self, input_shape, number_classes):
        
        print ("running model 7")

        model = Sequential()
        
        model.add(Conv1D(filters=64, kernel_size=3, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Conv1D(filters=64, kernel_size=3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling1D(pool_size=2))
                   
        model.add(Conv1D(filters=128, kernel_size=3))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        model.add(Conv1D(filters=128, kernel_size=3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(AveragePooling1D(pool_size=2))
    
        model.add(Conv1D(filters=128, kernel_size=3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(AveragePooling1D(pool_size=2))
        #model.add(Dropout(.5))
        
        model.add(Conv1D(filters=128, kernel_size=3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        
        #model.add(MaxPooling1D(pool_size=2))
        model.add(AveragePooling1D(pool_size=2))
        model.add(LSTM(128, return_sequences=True))   
        #model.add(LSTM(128, return_sequences=False)) 
        model.add(BatchNormalization())
        #model.add(Dropout(.5))
        
        model.add(Flatten()) 
        #model.add(BatchNormalization())
        
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.5))
           
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
                            
        model.add(Dense(number_classes, activation='softmax'))  
        model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        return model
    
    def ensure_no_leakage_train_val(self, df_train_test):
        """
        Ensure no data leakage in train/validation split:
        1) All subsequences of a sequence are in either train or val set.
        2) None of the subsequences of validation sequences are in the train set.
        3) None of the subsequences of reverse complements of validation sequences are in the train set or test_set
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

    def train_valid_split(self, df_train_test):
            # Get unique sequence IDs from df_train_test
        unique_train_ids = df_train_test['id'].unique()

        # Split unique_train_ids into train and validation
        train_ids, val_ids = train_test_split(
            unique_train_ids, test_size=0.2, stratify=df_train_test.drop_duplicates(subset=['id'])['target'], random_state=42
        )

        # Filter df_train_test based on the split train_ids and val_ids
        df_train_split = df_train_test[df_train_test['id'].isin(train_ids)]
        df_val_split = df_train_test[df_train_test['id'].isin(val_ids)]
        
        return df_train_split, df_val_split

    def train_model(self, df_train_test, target_to_index) :
        # Filter df_train_test based on the split train_ids and val_ids
        df_train_split, df_val_split = self.train_valid_split(df_train_test)
        # Prepare training data
        x_train = np.array([seq for seq in df_train_split['seq']])
        y_train = df_train_split['target'].values
        
        # Prepare validation data
        x_val = np.array([seq for seq in df_val_split['seq']])
        y_val = df_val_split['target'].values
   
        insh = (x_train.shape[1], 4)  # Input shape for one-hot encoded data
        model = self.build_model(insh, len(target_to_index))

        # Compute class weights
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.config['es_patience'], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-10)

        # Train the model using the training/validation split (validation is used for tuning)
        hist = model.fit(
            x=x_train, y=y_train,
            validation_data=(x_val, y_val),  # Use validation data here for tuning
            epochs=self.config['epochs'],
            callbacks=[early_stopping, reduce_lr],
            batch_size=self.config['batch_size'],
            verbose=1,
            class_weight=class_weights_dict
        )
        
        return model, hist

    def train_test_cv(self, dp):
        foldid = 1
        k_folds = self.config.get('k_folds', 5)
        
        print ("fold = ", foldid)
        for df_train_test, df_test in dp.get_KFold_Samples(n_splits=k_folds):
            
            if foldid < -1 :
                print ("skip foldid ", foldid )
                foldid = foldid + 1
                continue
            
            print()
            # Get unique sequence IDs from df_train_test
            unique_train_ids = df_train_test['id'].unique()
    
            # Split unique_train_ids into train and validation
            train_ids, val_ids = train_test_split(
                unique_train_ids, test_size=0.2, stratify=df_train_test.drop_duplicates(subset=['id'])['target'], random_state=42
            )
    
            # Filter df_train_test based on the split train_ids and val_ids
            df_train_split = df_train_test[df_train_test['id'].isin(train_ids)]
            df_val_split = df_train_test[df_train_test['id'].isin(val_ids)]
    
    
            print ("Train ids: ", len(df_train_split['id'].unique()), df_train_split['id'].unique())        
            print ("valid ids: ", len(df_val_split['id'].unique()), df_val_split['id'].unique())
            print ("Test ids: ", len(df_test['id'].unique()),df_test['id'].unique())
            
            # Prepare training data
            
            x_train = np.array([seq for seq in df_train_split['seq']])
            y_train = df_train_split['target'].values
            
            # Prepare validation data
            x_val = np.array([seq for seq in df_val_split['seq']])
            y_val = df_val_split['target'].values
    
            # Prepare test data (this won't be used for tuning, only for final evaluation)
            x_test = np.array([seq for seq in df_test['seq']])
            y_test = df_test['target'].values
    
            insh = (x_train.shape[1], 4)  # Input shape for one-hot encoded data
            model = self.build_model(insh, len(dp.terget_to_index))
    
            # Compute class weights
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weights_dict = dict(enumerate(class_weights))
    
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=self.config['es_patience'], restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-10)
    
            # Train the model using the training/validation split (validation is used for tuning)
            hist = model.fit(
                x=x_train, y=y_train,
                validation_data=(x_val, y_val),  # Use validation data here for tuning
                epochs=self.config['epochs'],
                callbacks=[early_stopping, reduce_lr],
                batch_size=self.config['batch_size'],
                verbose=1,
                class_weight=class_weights_dict
            )
              
            # Store the evaluation results
            #self.measures.update(model, df_test, hist, self.config, dp, foldid=foldid)
            self.measures.update(model, df_test, hist, self.config, foldid=foldid)
            del model
            gc.collect()
            
            
            with open(f'hist_{dp.classifier_name}_fold_{foldid}.pkl', 'wb') as file:
                pickle.dump(hist.history, file)
    
            print("================================================")
            print("================================================")
            print(self.measures.df)
            print("accuracy_sv: " ,  self.measures.df['accuracy_sv'] .mean())
            print("recall_sv: " ,  self.measures.df['recall_sv'] .mean())
            print("precision_sv: " ,  self.measures.df['precision_sv'] .mean())
            print("f1_sv: " ,  self.measures.df['f1_sv'] .mean())
            
            self.measures.write_measures()
            foldid = foldid + 1
    
    def encode_family_within_group(self, group):
        unique_families = group.unique()  # Get unique families from the Series (group)
        family_mapping = {family: index for index, family in enumerate(unique_families)}  # Map to index
        return group.map(family_mapping), family_mapping  # Return the mapped values as a series

    def encode_family_within_group(self, df):
        unique_families = df['family'].unique()  # Get unique families in the group
        family_mapping = {family: index for index, family in enumerate(unique_families)}  # Map to index
        df['family_encode'] = df['family'].map(family_mapping)  # Apply encoding to 'family' column
        return df, family_mapping  # Return the DataFrame and the mapping

    def plots(self, history):
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.show()
        input("Press Enter to continue...")
    
    def train_heirarchy_cv(self, dp):
        foldid = 1
        k_folds = self.config.get('k_folds', 5)
        
        print ("fold = ", foldid)
        for df_train_test, df_test in dp.get_KFold_Samples(n_splits=k_folds):

            if foldid < -1 :
                print ("skip foldid ", foldid )
                foldid = foldid + 1
                continue
            
            print()
                                   
            family_mapping = {} 
            dd = df_train_test.groupby('target')['family'].unique()
            family_mapping[0] = {family: index for index, family in enumerate(dd[0])}
            family_mapping[1] = {family: index for index, family in enumerate(dd[1])}
            
            print ("Training model: order ")
            
            model, hist = self.train_model(df_train_test, dp.terget_to_index)
            #with open('history.pkl', 'wb') as file:
                #pickle.dump(hist, file)
            
            #self.plots(hist)
            model.save('class_model.h5')  # .h5 is one option (HDF5 format)
            del model
            gc.collect()  
            
            for order in df_train_test['target'].unique():
                df_train_test_order = df_train_test[df_train_test['target'] == order].copy()                
                df_train_test_order['target'] = df_train_test_order['family'].map(family_mapping[order])
                print (f"Training model: family : {order} ")
                model, hist = self.train_model(df_train_test_order, family_mapping[order])
                
                #self.plots(hist)
                
                model.save(f'order_{order}_model.h5')  # .h5 is one option (HDF5 format)
                #with open(f'order_{order}_hist.h5', 'wb') as file:
                    #pickle.dump(hist, file)
                    
                del model
                gc.collect()
            
            encoded_dfs = []
            for order in df_test['target'].unique():
                df_test_order = df_test[df_test['target'] == order].copy()                
                df_test_order['family_encoded'] = df_test_order['family'].map(family_mapping[order])
                encoded_dfs.append(df_test_order)
                          
            df_test = pd.concat(encoded_dfs, ignore_index=True)
            df_test['order_encoded'] = df_test['target']
            self.measures.evaluate_hierarchical_models_v5(foldid, df_test)
            # Store the evaluation results
            #self.measures.update(model, df_test, hist, self.config, foldid=foldid)
            #del model
            #gc.collect()
    
            #print("================================================")
            #print("================================================")
            #print(self.measures.df)           
            #self.measures.write_measures()
            foldid = foldid + 1
            
    def split_train_valid(self, df_train_test) :
        # Get unique sequence IDs from df_train_test
        unique_train_ids = df_train_test['id'].unique()

        # Split unique_train_ids into train and validation
        train_ids, val_ids = train_test_split(
            unique_train_ids, test_size=0.2, stratify=df_train_test.drop_duplicates(subset=['id'])['target'], random_state=42
        )

        # Filter df_train_test based on the split train_ids and val_ids
        df_train_split = df_train_test[df_train_test['id'].isin(train_ids)]
        df_val_split = df_train_test[df_train_test['id'].isin(val_ids)]
        
        return df_train_split, df_val_split
           

