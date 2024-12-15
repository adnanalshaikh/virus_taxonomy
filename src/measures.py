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
        """Initialize the Measures class for tracking and saving model metrics."""
        self.dp = dp
        self.node_dir = HOME_DIR / "results" / f"{dp.classifier_name}"
        self.node_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Define filename and create DataFrames for measures
        self.filename = filename or f"{dp.classifier_name}.csv" 
        
        # Define the columns we want to track
        header = [
            'classifier', 'model', 'fold', 'num_epochs', 'batch_size', 'learning_rate',
            'loss', 'accuracy', 'precision', 'recall', 'f1',
            'accuracy_sv', 'precision_sv', 'recall_sv', 'f1_sv', 'roc_auc', 'report', 'conf_matrix'
        ]
        self.df = pd.DataFrame(columns=header) if filename is None else pd.read_csv(filename)
        
        # heirarchy measures 
        self.h_header = ['fold_id', 'h_accuracy', 'h_precision', 'h_recall', 'h_f1_score']
        self.hdf = pd.DataFrame(columns=self.h_header)

        # make a reults directory if not exist 


    
    def voting_predict(self, model, df_test):
        """Perform soft voting predictions for all sequences."""
        
        ids = []  # To store sequence IDs
        probs = []  # To store soft voting probabilities
        predictions = []  # To store predicted classes (np.argmax(sv))
    
        seqs_ids = df_test['id'].unique()
        for seq_id in seqs_ids:
            df_filtered = df_test[df_test['id'] == seq_id]
            x_test_seq = np.array([list(seq) for seq in df_filtered['seq']])
            
            # Predict for all subsequences
            y_pred = model.predict(x_test_seq, verbose=0)
            
            # Soft voting: average the probabilities
            sv = np.mean(y_pred, axis=0)
            y_sv = np.argmax(sv)  # Soft voting result (predicted class)
    
            ids.append(seq_id)
            probs.append(sv)
            predictions.append(y_sv)
        
        # Convert lists to NumPy arrays
        ids = np.array(ids)
        predictions = np.array(predictions)  # Classes as NumPy array
        combined = np.column_stack((ids, predictions))  # Combine IDs and classes into a 2-column NumPy array
        
        # Create a pandas DataFrame for IDs and probability lists
        df_probs = pd.DataFrame({'id': ids, 'sv': list(probs)})
    
        return combined, df_probs

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
            y_predict.append([seq_id, y_sv])
        
        return np.array(y_predict)


    def calculate_metrics(self, y_true, y_pred):
        """Calculate precision, recall, F1-score, and accuracy."""
        
        return {
            'precision': precision_score(y_true, y_pred, average='micro'),
            'recall': recall_score(y_true, y_pred, average='micro'),
            'f1': f1_score(y_true, y_pred, average='micro'),
        }

    def calculate_soft_voting_metrics(self, y_pred_sv, df_test, target_names):
        """Calculate soft voting metrics."""
        y_pred = y_pred_sv[:, 1]
        y_true = [df_test[df_test['id'] == seq_id]['target'].iloc[0] for seq_id in y_pred_sv[:, 0]]

        return {
            'accuracy_sv': accuracy_score(y_true, y_pred),
            'precision_sv': precision_score(y_true, y_pred, average='micro'),
            'recall_sv': recall_score(y_true, y_pred, average='micro'),
            'f1_sv': f1_score(y_true, y_pred, average='micro'),
            'report' : classification_report(y_true, y_pred, target_names=target_names),
            'conf_matrix' : confusion_matrix(y_true, y_pred)
        }

    def calculate_roc_auc(self, df_test, df_pred_probs):
        """
        Calculate ROC AUC, handling multi-class and binary classification cases.
        
        Parameters:
        - df_test: DataFrame containing 'id' and 'target' columns (true labels).
        - df_pred_probs: DataFrame containing 'id' and 'sv' columns (predicted probabilities).
        
        Returns:
        - ROC AUC score (float) or None if an error occurs.
        """
        try:
            # Ensure unique rows for 'id' in df_test and align with df_pred_probs
            unique_df = df_test.drop_duplicates(subset='id', keep='last')[['id', 'target']]
            aligned_df = pd.merge(unique_df, df_pred_probs, on="id")
    
            # Extract true labels (y_true) and predicted probabilities (y_pred_probs)
            y_true = aligned_df["target"].to_numpy()
            y_pred_probs = np.array(aligned_df["sv"].tolist())
    
            # Check for multi-class or binary classification
            if len(np.unique(y_true)) > 2:  # Multi-class
                return roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
            else:  # Binary
                return roc_auc_score(y_true, y_pred_probs[:, 1])
        except ValueError as e:
            print(f"Error calculating ROC AUC: {e}")
            return None


    def update(self, model, df_test, hist, config, foldid=0):
        """Update measures DataFrame with metrics for a given model and fold."""
        
        print(f"Evaluating model for fold {foldid}...")
        
        # Prepare test data
        x_test = np.array([list(seq) for seq in df_test['seq']])
        y_test = df_test['target'].values  # Should be 1D, like [0, 1, 2, ...]
    
        # Initialize row with hyperparameters and metrics
        row = {
            'classifier' : self.dp.classifier_name,
            'model': config['modelid'],
            'fold': foldid,
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'es_patience': config['es_patience'],
            'num_epochs': len(hist.history['loss'])
        }
        
        # Evaluate model on subsequences of the test set without voting
        row['loss'], row['accuracy'] = model.evaluate(x_test, y_test, verbose=0)
        
        # Predictions and metrics
        y_pred_probs = model.predict(x_test)  # Predict probabilities, shape (n_samples, n_classes)
        y_pred_class = np.argmax(y_pred_probs, axis=1)
        
        # Standard metrics (micro average)
        row.update(self.calculate_metrics(y_test, y_pred_class))
  
        # Soft voting metrics & classification report & confusion matrix
        y_pred_sv, df_pred_probs = self.voting_predict(model, df_test)
        row.update(self.calculate_soft_voting_metrics(y_pred_sv, df_test, self.dp.target_to_index.keys()))
        
        row['roc_auc'] = self.calculate_roc_auc(df_test, df_pred_probs)

        # Append results
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.write_measures()

        #disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Alphaflexiviridae', 'Betaflexiviridae', 'Tymoviridae'])
        #disp.plot(cmap='viridis')  # Optional: Customize the color map
        #plt.title("Confusion Matrix")
        #plt.show()
    
        #print("============================================")
        #print(f'{self.dp.classifier_name}_classifier_fold_{foldid} results\n')
        #print("Report:\n")
        #print(report)
        #print("Confusion matrix:\n")
        #print(conf_matrix)

    def update1(self, model, df_test, hist, config, foldid=0):
        """Update measures DataFrame with metrics for a given model and fold."""
        print(f"Evaluating model for fold {foldid}...")

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


    def evaluate_hierarchical_model_predictions(self, foldid, df_test):
        """
        Evaluates hierarchical models M1, M2, and M3 on the given test data.
    
        Parameters:
        - foldid (int): Identifier for the fold being evaluated
        - df_test (pd.DataFrame): Test dataset with columns 'id', 'seq', 'order_encoded', 'family_encoded'
    
        Returns:
        - dict: A dictionary containing accuracy, F1-score, recall, and precision
        """
        
        # Load models
        model_m1 = load_model('class_model.h5')
        model_m2 = load_model('order_0_model.h5')
        model_m3 = load_model('order_1_model.h5')
    
        # Prepare results DataFrame
        df_results = df_test.drop_duplicates(subset='id').reset_index(drop=True)
        df_results = df_results.drop(columns=['seq', 'is_complement_of', 'target'], errors='ignore')
        df_results.rename(columns={'family_encoded': 'true_family', 'order_encoded': 'true_order'}, inplace=True)
    
        # Step 1: Predict orders using Model M1
        y_pred_order = self.voting_predict_heirarchy(model_m1, df_test)
        df_results_order = pd.DataFrame(y_pred_order, columns=['id', 'pred_order'])
        df_results = pd.merge(df_results, df_results_order, on='id', how='left')
    
        # Filter correct orders
        correct_order_ids = df_results[df_results['pred_order'] == df_results['true_order']]['id']
        df_correct_order = df_test[df_test['id'].isin(correct_order_ids)]
    
        # Separate samples by order
        df_correct_order_0 = df_correct_order[df_correct_order['order_encoded'] == 0].copy()
        df_correct_order_1 = df_correct_order[df_correct_order['order_encoded'] == 1].copy()
    
        # Step 2: Predict families for correct orders
        if not df_correct_order_0.empty:
            y_pred_family_0 = self.voting_predict_heirarchy(model_m2, df_correct_order_0)
            df_results_family_0 = pd.DataFrame(y_pred_family_0, columns=['id', 'pred_family'])
            df_results = pd.merge(df_results, df_results_family_0, on='id', how='left')
            df_results.fillna(-1, inplace=True)
    
        if not df_correct_order_1.empty:
            y_pred_family_1 = self.voting_predict_heirarchy(model_m3, df_correct_order_1)
            df_results_family_1 = pd.DataFrame(y_pred_family_1, columns=['id', 'pred_family_1'])
            df_results = pd.merge(df_results, df_results_family_1, on='id', how='left')
            df_results['pred_family'] = df_results['pred_family_1'].combine_first(df_results['pred_family'])
            df_results.drop(columns=['pred_family_1'], inplace=True)
    
        # Step 3: Calculate evaluation metrics
        accuracy = accuracy_score(df_results['true_family'], df_results['pred_family'])
        f1 = f1_score(df_results['true_family'], df_results['pred_family'], average='weighted', zero_division=0)
        recall = recall_score(df_results['true_family'], df_results['pred_family'], average='weighted', zero_division=0)
        precision = precision_score(df_results['true_family'], df_results['pred_family'], average='weighted', zero_division=0)
    
        # Compile results
        results = {
            "fold_id": foldid,
            "h_accuracy": accuracy,
            "h_precision": precision,
            "h_recall": recall,
            "h_f1_score": f1
        }
    
        # Log results
        self.hdf = pd.concat([self.hdf, pd.DataFrame([results])], ignore_index=True)
        self.write_measures(hierarchy=True)
        print(self.hdf)
    
        return results

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
        
        y_pred_order_sv = self.voting_predict_heirarchy(model_m1, df_test)
        
        df_results_order = pd.DataFrame(y_pred_order_sv, columns = ['id', 'pred_order'])
        df_results = pd.merge(df_results, df_results_order[['id', 'pred_order']], on='id', how='left')
                
        df_correct_order_ids = df_results[df_results['pred_order'] == df_results['true_order']]['id']
        
        df_correct_order = df_test[df_test['id'].isin(df_correct_order_ids)]
        
        df_correct_order_0 = df_correct_order[df_correct_order['order_encoded'] == 0].copy() 
        df_correct_order_1 = df_correct_order[df_correct_order['order_encoded'] == 1].copy() 
        
        # Step 5: Pass the correct samples to M2 and M3
        if len(df_correct_order_0) > 0:
            y_pred_family_sv = self.voting_predict_heirarchy(model_m2, df_correct_order_0)
            df_results_order_0 = pd.DataFrame(y_pred_family_sv, columns = ['id', 'pred_family'])
            df_results = pd.merge(df_results, df_results_order_0[['id', 'pred_family']], on='id', how='left')
            df_results.fillna(-1, inplace=True)
           
        if len(df_correct_order_1) > 0:
            y_pred_family_sv = self.voting_predict_heirarchy(model_m3, df_correct_order_1)
            df_results_order_1 = pd.DataFrame(y_pred_family_sv, columns = ['id', 'pred_family_1'])
            df_results = pd.merge(df_results, df_results_order_1[['id', 'pred_family_1']], on='id', how='left')
            df_results['pred_family'] = df_results['pred_family_1'].combine_first(df_results['pred_family'])
            df_results.drop(columns = ['pred_family_1'], inplace = True)
        
        accuracy = accuracy_score(df_results['true_family'], df_results['pred_family'])
        f1 = f1_score(df_results['true_family'], df_results['pred_family'], average='weighted', zero_division=0)
        recall = recall_score(df_results['true_family'], df_results['pred_family'], average='weighted', zero_division=0)
        precision = precision_score(df_results['true_family'], df_results['pred_family'], average='weighted', zero_division=0)
        
        # Compile results
        results = {
            "fold_id": foldid,
            "h_accuracy": accuracy,
            "h_precision": precision,
            "h_recall": recall,
            "h_f1_score": f1
        }
    
        # Log results
        self.hdf = pd.concat([self.hdf, pd.DataFrame([results])], ignore_index=True)
        self.write_measures(hierarchy=True)
        print(self.hdf)

        return results

    def write_measures(self, hierarchy=False):
        """
        Writes the appropriate DataFrame to a CSV file based on the hierarchy flag.
    
        Parameters:
        - hierarchy (bool): If True, writes hierarchical DataFrame (hdf) with a prefixed filename.
                            If False, writes the standard DataFrame (df).
        """
        # Determine the filename based on hierarchy flag
        output_filename = f"h_{self.filename}" if hierarchy else self.filename
    
        # Select the correct DataFrame to write
        data_to_save = self.hdf if hierarchy else self.df
    
        # Write to CSV file
        data_to_save.to_csv(self.node_dir / output_filename, index=False)

