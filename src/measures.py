from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, multilabel_confusion_matrix
)
import numpy as np
import pandas as pd
from scipy.stats import mode
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tensorflow.keras.models import load_model

class Measures:
    def __init__(self, filename=None):
        self.filename = filename or "measures.csv"
        
        # Define the columns we want to track
        header = [
            'model', 'fold', 'num_epochs', 'batch_size', 'learning_rate',
            'loss', 'accuracy', 'precision', 'recall', 'f1',
            'accuracy_sv', 'precision_sv', 'recall_sv', 'f1_sv', 'roc_auc'
        ]
        if filename is None:
            self.df = pd.DataFrame(columns=header)
        else:
            self.df = pd.read_csv(filename)

    def voting_predict(self, model, df_test):
        y_predict = []
        seqs_ids = df_test['id'].unique()
        for seq_id in seqs_ids:
            df_filtered = df_test[df_test['id'] == seq_id]
            x_test_seq = np.array([list(seq) for seq in df_filtered['seq']])
            
            # Predict for all subsequences
            y_pred = model.predict(x_test_seq, verbose=0)
            
            # Soft voting: average the probabilities
            sv = np.mean(y_pred, axis=0)
            y_sv = np.argmax(sv)  # Soft voting result

            # Store the soft voting result for the whole sequence
            y_predict.append([seq_id, y_sv])
        
        return np.array(y_predict)

    def voting_predict_heirarchy(self, model, df_test):
        y_predict = []
        seqs_ids = df_test['id'].unique()
        for seq_id in seqs_ids:
            df_filtered = df_test[df_test['id'] == seq_id]
            x_test_seq = np.array([list(seq) for seq in df_filtered['seq']])
            
            # Predict for all subsequences
            y_pred = model.predict(x_test_seq, verbose=0)
            
            # Soft voting: average the probabilities
            sv = np.mean(y_pred, axis=0)
            y_sv = np.argmax(sv)  # Soft voting result

            # Store the soft voting result for the whole sequence
            y_true = df_filtered.iloc[0]['order_encoded']
            y_predict.append([seq_id, y_sv, y_true])
        
        return np.array(y_predict)
    
    def update(self, model, df_test, hist, config, foldid=0):
        # Prepare test data
        x_test = np.array([list(seq) for seq in df_test['seq']])
        
        # Ensure y_test is a 1D array of class labels, not one-hot encoded
        y_test = df_test['target'].values  # Should be 1D, like [0, 1, 2, ...]
    
        # Initialize row with hyperparameters and metrics
        row = {
            'model': config['modelid'],
            'fold': foldid,
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'num_epochs': len(hist.history['loss'])
        }
        
        # Evaluate model on test set
        row['loss'], row['accuracy'] = model.evaluate(x_test, y_test, verbose=0)
        
        # Predict class probabilities
        y_pred_probs = model.predict(x_test)  # Predict probabilities, shape (n_samples, n_classes)
        
        # Convert predicted probabilities to predicted classes
        y_pred_class = np.argmax(y_pred_probs, axis=1)
        
        # Standard metrics (micro average)
        row['f1'] = f1_score(y_test, y_pred_class, average='micro')
        row['precision'] = precision_score(y_test, y_pred_class, average='micro')
        row['recall'] = recall_score(y_test, y_pred_class, average='micro')
    
        # Soft voting metrics
        y_pred_sv = self.voting_predict(model, df_test)
        
        y_pred_test_sv = []
        for i in range(len(y_pred_sv)):
            target_label = df_test[df_test['id'] == y_pred_sv[i, 0]]['target'].iloc[0]
            y_pred_test_sv.append([y_pred_sv[i, 0], y_pred_sv[i, 1], target_label])
        
        y_pred_test_sv = np.array(y_pred_test_sv)
        
        row['f1_sv'] = f1_score(y_pred_test_sv[:, 2], y_pred_test_sv[:, 1], average='micro')
        row['precision_sv'] = precision_score(y_pred_test_sv[:, 2], y_pred_test_sv[:, 1], average='micro')
        row['recall_sv'] = recall_score(y_pred_test_sv[:, 2], y_pred_test_sv[:, 1], average='micro')
        row['accuracy_sv'] = accuracy_score(y_pred_test_sv[:, 2], y_pred_test_sv[:, 1])
        
        # Compute ROC AUC (for multi-class classification)
        try:
            row['roc_auc'] = roc_auc_score(y_test, y_pred_probs, multi_class="ovr", average="macro")
        except ValueError as e:
            print(f"Error calculating ROC AUC: {e}")
            row['roc_auc'] = None  # Set to None or handle the error appropriately
    
        # Append row to the DataFrame
        self.df = self.df.append(row, ignore_index=True)
        
        # Print out the metrics
        for k, v in row.items():
            print(f"{k}: {v}")


    def evaluate_hierarchical_models_v5(self, df_test):
        """
        Evaluates hierarchical models M1, M2, and M3 on the given test data.
    
        Parameters:
        - test_data (pd.DataFrame): Test dataset with columns 'id', 'seq', 'order', 'family'
        - m1_path (str): Path to the trained Model M1
        - m2_path (str): Path to the trained Model M2
        - m3_path (str): Path to the trained Model M3
    
        Returns:
        - A dictionary containing accuracy, f1, recall, and precision
        """
        
        # Load models
        model_m1 = load_model('class_model.h5')
        model_m2 = load_model('order_0_model.h5')
        model_m3 = load_model('order_1_model.h5')
        
        # Extract features and true labels
        #X_test = np.array(list(test_data['seq'].values))  # Sequences as array
        #y_true_order = test_data['order_encoded'].values  # True order labels
        #y_true_family = test_data['family_encoded'].values  # True family labels
        ########################################################################
        
        # Step 1: Use Model M1 to predict the order for all test samples
        # Soft voting metrics
        df_results = df_test.drop_duplicates(subset='id').reset_index()
        df_results = df_results.drop(columns=['seq', 'is_complement_of', 'target'])
        df_results.rename(columns = { 'family_encoded' : 'true_family', 'order_encoded' : 'true_order'}, inplace = True)
        
        y_pred_order_sv = self.voting_predict(model_m1, df_test)
        df_results_order = pd.DataFrame(y_pred_order_sv, columns = ['id', 'pred_order'])
        df_results = pd.merge(df_results, df_results_order[['id', 'pred_order']], on='id', how='left')
                
        df_correct_order_ids = df_results[df_results['pred_order'] == df_results['true_order']]['id']
        
        df_correct_order = df_test[df_test['id'].isin(df_correct_order_ids)]
        
        df_correct_order_0 = df_correct_order[df_correct_order['order_encoded'] == 0].copy() 
        df_correct_order_1 = df_correct_order[df_correct_order['order_encoded'] == 1].copy() 
        
        # Step 5: Pass the correct samples to M2 and M3
        if len(df_correct_order_0) > 0:
            y_pred_family_sv = self.voting_predict(model_m2, df_correct_order_0)
            df_results_order_0 = pd.DataFrame(y_pred_family_sv, columns = ['id', 'pred_family'])
            df_results = pd.merge(df_results, df_results_order_0[['id', 'pred_family']], on='id', how='left')
            df_results.fillna(-1, inplace=True)
           
        if len(df_correct_order_1) > 0:
            y_pred_family_sv = self.voting_predict(model_m3, df_correct_order_1)
            df_results_order_1 = pd.DataFrame(y_pred_family_sv, columns = ['id', 'pred_family_1'])
            df_results = pd.merge(df_results, df_results_order_1[['id', 'pred_family_1']], on='id', how='left')
            df_results['pred_family'] = df_results['pred_family_1'].combine_first(df_results['pred_family'])
            df_results.drop(columns = ['pred_family_1'], inplace = True)
        accuracy = accuracy_score(df_results['true_family'], df_results['pred_family'])
        
        f1 = f1_score(df_results['true_family'], df_results['pred_family'], average='weighted', zero_division=0)
        recall = recall_score(df_results['true_family'], df_results['pred_family'], average='weighted', zero_division=0)
        precision = precision_score(df_results['true_family'], df_results['pred_family'], average='weighted', zero_division=0)
        
        # Return the results
        results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "precision": precision
        }
        
        print (results)
        
        return results


    def evaluate_hierarchical_models_v5__(self, test_data):
        """
        Evaluates hierarchical models M1, M2, and M3 on the given test data.
    
        Parameters:
        - test_data (pd.DataFrame): Test dataset with columns 'id', 'seq', 'order', 'family'
        - m1_path (str): Path to the trained Model M1
        - m2_path (str): Path to the trained Model M2
        - m3_path (str): Path to the trained Model M3
    
        Returns:
        - A dictionary containing accuracy, f1, recall, and precision
        """
        
        # Load models
        model_m1 = load_model('class_model.h5')
        model_m2 = load_model('order_0_model.h5')
        model_m3 = load_model('order_1_model.h5')
        
        # Extract features and true labels
        X_test = np.array(list(test_data['seq'].values))  # Sequences as array
        y_true_order = test_data['order_encoded'].values  # True order labels
        y_true_family = test_data['family_encoded'].values  # True family labels
    
        # Step 1: Use Model M1 to predict the order for all test samples
        y_pred_order = model_m1.predict(X_test)
        y_pred_order = np.argmax(y_pred_order, axis=1)  # Assuming softmax output
        
        # Step 2: Initialize family predictions (mark incorrect order predictions as invalid family)
        final_family_predictions = np.full_like(y_true_family, -1)  # Default to -1 for incorrect order predictions
        
        # Step 3: Identify correctly predicted orders
        correct_order_indices = np.where(y_pred_order == y_true_order)[0]  # Indices where M1's prediction is correct
        
        # Step 4: Slice the test data for M2 and M3 based on correct predictions
        m2_indices = np.where(y_pred_order[correct_order_indices] == 0)[0]  # Correct samples predicted for M2
        m3_indices = np.where(y_pred_order[correct_order_indices] == 1)[0]  # Correct samples predicted for M3
        
        # Step 5: Pass the correct samples to M2 and M3
        if len(m2_indices) > 0:
            X_test_m2 = X_test[correct_order_indices][m2_indices]
            m2_predictions = model_m2.predict(X_test_m2)
            final_family_predictions[correct_order_indices[m2_indices]] = np.argmax(m2_predictions, axis=1)
    
        if len(m3_indices) > 0:
            X_test_m3 = X_test[correct_order_indices][m3_indices]
            m3_predictions = model_m3.predict(X_test_m3)
            final_family_predictions[correct_order_indices[m3_indices]] = np.argmax(m3_predictions, axis=1)
        
        # Step 6: Calculate metrics (consider all predictions, including the incorrect ones marked as -1)
        accuracy = accuracy_score(y_true_family, final_family_predictions)
        f1 = f1_score(y_true_family, final_family_predictions, average='weighted', zero_division=0)
        recall = recall_score(y_true_family, final_family_predictions, average='weighted', zero_division=0)
        precision = precision_score(y_true_family, final_family_predictions, average='weighted', zero_division=0)
        
        # Return the results
        results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "precision": precision
        }
        
        return results


    def get_measures(self):
        return self.df

    def write_measures(self):
        self.df.to_csv(self.filename, index=False)

    def eval(self):
        """
        This function can be further extended based on the problem specifics.
        """
        pass
