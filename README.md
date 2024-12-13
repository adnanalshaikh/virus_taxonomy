# Hierarchical Classification of Viral DNA Sequences Using a CNN-LSTM Framework

This repository contains the code accompanying the paper:
*"Hierarchical Classification of Viral DNA Sequences Using a CNN-LSTM Framework"* by **Adnan Salman** and **Naeel Khuffash**.

## Overview
This repository implements the methodology described in the paper to demonstrate the feasibility and results of the proposed hierarchical classification approach for viral DNA sequences.

### **Methodology**
We employed a hierarchical learning framework to classify the taxonomy of a virus based on its DNA sequence. The process involves:

1. **Order Classification**:
   - An initial classifier predicts the viral order.
2. **Family Classification**:
   - Based on the predicted order, a second classifier identifies the viral family.

While further classifiers could predict the genus and species, this research focuses on evaluating the performance of the order and family classifiers.

### **Key Constraints**
- **Viral Orders**: The analysis is limited to two orders, *Martellivirales* and *Tymovirales*.
- **Data Filtering**:
  - Families with fewer than five species were excluded.
  - Species with sequence lengths shorter than 100 base pairs were removed to maintain robust classification performance.

### **Significance**
This hierarchical classification approach enables accurate viral taxonomy from DNA sequences. By breaking down the classification process into smaller, targeted steps, we simplify the complexity of viral taxonomy, enhancing the ability to study and address viral threats.

### **Architecture**
![Architecture Diagram](https://github.com/user-attachments/assets/6cae7b69-c875-4e23-916c-940f574fc678)

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/adnanalshaikh/virus_taxonomy
cd virus_taxonomy
```
### **2. Install the required dependency**
```bash
pip install -r requirements.txt
```

## **Directory Structure**
```bash
virus_taxonomy/
├── src/                        # Main source code
│   ├── dna_taxonomy.py         # Entry point for running the pipeline
│   ├── cnn_model.py            # CNN-LSTM model implementation
│   ├── measures.py             # Functions for performance evaluation
│   ├── preprocess.py           # Preprocessing functions
│   ├── dna_data_loader.py      # Utilities for data retrieval from NCBI
├── data/                       # Input data
│   ├── Tymovirales_refseq.csv  # Extracted data using dna_data_loader.py
│   ├── Martellivirales_refseq.csv # Extracted data using dna_data_loader.py
│   ├── Alsuviricetes_refseq.csv   # Extracted data using dna_data_loader.py
│   ├── VMR_MSL39_v1.xlsx          # Virus metadata file downloaded from ICTV (https://ictv.global/vmr)
├── results/                    # Directory for outputs
├── requirements.txt            # Dependencies for the project
└── README.md                   # Documentation
```

## **Citation**
If you use this code or methodology in your work, please cite the following paper:
Salman, A., & Khuffash, N. (2024). Hierarchical Classification of Viral DNA Sequences Using a CNN-LSTM Framework.

## **Resources**

- **Virus Metadata File (VMR_MSL39_v1.xlsx)**:
  - The metadata file used in this project is downloaded from the [ICTV Virus Metadata Resource](https://ictv.global/vmr).
  - Visit the [ICTV VMR Page](https://ictv.global/vmr) for more details and updates.
