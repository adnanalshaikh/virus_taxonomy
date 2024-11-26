from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

HOME_DIR = Path(__file__).resolve().parent.parent

class Measures:
    def __init__(self, dp, filename=None):
        self.dp = dp
        self.node_dir = HOME_DIR / "data" / f"{dp.classifier_name}"
        self.node_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.filename = filename or "measures.csv"
        
        # Define the columns we want to track
        header = [
            'classifier', 'model', 'fold', 'num_epochs', 'batch_size', 'learning_rate',
            'loss', 'accuracy', 'precision', 'recall', 'f1',
            'accuracy_sv', 'precision_sv', 'recall_sv', 'f1_sv', 'roc_auc'
        ]
        if filename is None:
            self.df = pd.DataFrame(columns=header)
        else:
            self.df = pd.read_csv(filename)
        
        # heirarchy measures 
        self.h_header = ['foldid', 'h_accuracy', 'h_precision', 'h_recall', 'h_f1']
        self.hdf = pd.DataFrame(columns=self.h_header)

        # make a reults directory if not exist 

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
        
        #create directory in the data directory to save all data related to this taxa 
        dp = self.dp
        # Prepare test data
        x_test = np.array([list(seq) for seq in df_test['seq']])
        y_test = df_test['target'].values  # Should be 1D, like [0, 1, 2, ...]
    
        # Initialize row with hyperparameters and metrics
        row = {
            'classifier' : dp.classifier_name,
            'model': config['modelid'],
            'fold': foldid,
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'es_patience': config['es_patience'],
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
        y_pred = y_pred_test_sv[:, 1]
        y_true = y_pred_test_sv[:, 2]
        
        row['f1_sv'] = f1_score(y_true, y_pred, average='micro')
        row['precision_sv'] = precision_score(y_true, y_pred, average='micro')
        row['recall_sv'] = recall_score(y_true, y_pred, average='micro')
        row['accuracy_sv'] = accuracy_score(y_true, y_pred)
        
        class_names = dp.terget_to_index.keys()
        report = classification_report(y_true, y_pred, target_names=class_names)
        class_report_fn = self.node_dir / f'{dp.classifier_name}_class_report_{foldid}.txt'
        with open(class_report_fn, 'w') as file:
            file.write(report)
    
        conf_matrix = confusion_matrix(y_true, y_pred)
        conf_matrix_fn = self.node_dir / f'{dp.classifier_name}_conf_matrix_{foldid}.npy'
        np.save(conf_matrix_fn, conf_matrix)
        
        
        #disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Alphaflexiviridae', 'Betaflexiviridae', 'Tymoviridae'])
        #disp.plot(cmap='viridis')  # Optional: Customize the color map
        #plt.title("Confusion Matrix")
        #plt.show()
    
        print("============================================")
        print(f'{dp.classifier_name}_classifier_fold_{foldid} results\n')
        print("Report:\n")
        print(report)
        print("Confusion matrix:\n")
        print(conf_matrix)
        
        try:
            # Check if it's multi-class or binary classification
            if len(set(y_test)) > 2:  # Multi-class
                roc_auc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr", average="macro")
            else:  # Binary classification
                roc_auc = roc_auc_score(y_test, y_pred_probs[:, 1])  # Use probabilities for the positive class
        except ValueError as e:
            print(f"Error calculating ROC AUC: {e}")
            roc_auc = None  # Handle error gracefully
        
        # Store the result
        row['roc_auc'] = roc_auc
        row['report'] = report
        row['conf_matrix'] = conf_matrix
        # Append row to the DataFrame
        #self.df = self.df.append(row, ignore_index=True)
        
        row_df = pd.DataFrame([row])  # Ensure row is wrapped in a list to create a DataFrame
        self.df = pd.concat([self.df, row_df], ignore_index=True)

        neasures_fn = self.node_dir / self.filename
        self.df.to_csv(neasures_fn, index=False)

    def evaluate_hierarchical_models_v5(self, foldid, df_test):
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
            "h_accuracy": accuracy,
            "h_f1_score": f1,
            "h_recall": recall,
            "h_precision": precision
        }
        
        # Create a new DataFrame for the row to append
        new_row = pd.DataFrame([dict(zip(self.h_header, [foldid, accuracy, precision, recall, f1]))])

        # Append using pd.concat
        self.hdf = pd.concat([self.hdf, new_row], ignore_index=True)

        filename = "h_" + self.filename
        self.hdf.to_csv(filename, index=False)
        print (self.hdf)
        
        return results


    def get_measures(self):
        return self.df

    def write_measures(self):
        self.df.to_csv(self.filename, index=False)

