# Hierarchical Classification of Viral DNA Sequences Using a CNN-LSTM Framework

This repository contains the code for the paper:
"Hierarchical Classification of Viral DNA Sequences Using a CNN-LSTM Framework" by Adnan Salman and Naeel Khuffash.

## Overview
This repository implements the methodology described in the paper to demonstrate the feasibility and results of the proposed hierarchical classification approach for viral DNA sequences.

We employed a hierarchical learning framework to classify the taxonomy of a virus based on its DNA sequence. The process involves:

- Order Classification: An initial classifier predicts the viral order.
- Family Classification: Based on the predicted order, a second classifier identifies the viral family.
While further classifiers could be applied to predict the genus and species of the virus, this research focuses on evaluating the performance of the order and family classifiers.

Key constraints in the study include:

- Viral Orders: The analysis is limited to two orders, Martellivirales and Tymovirales.
- Data Filtering: Families with fewer than five species and species with sequence lengths shorter than 100 base pairs were excluded to maintain robust classification performance.

This hierarchical classification approach provides accurate viral taxonomy from DNA sequences. By breaking down the classification into smaller, targeted steps, we simplify the complexity of viral taxonomy, enhancing the ability to study and address viral threats.

![archit](https://github.com/user-attachments/assets/6cae7b69-c875-4e23-916c-940f574fc678)

## Installation
Clone the repository:

```bash
git clone https://github.com/adnanalshaikh/virus_taxonomy
cd virus_taxonomy 
```

Install the required dependency:
```bash
pip install -r requirements.txt
```
## Directory Structure

```bash
virus_taxonomy/
├── src/               # Main source code
│   ├── main.py        # Entry point for running the pipeline
│   ├── model.py       # CNN-LSTM model implementation
│   ├── utils.py       # Utility functions
├── data/              # Sample input data
├── results/           # Directory for outputs
├── config.yaml        # Configuration file for experiments
├── requirements.txt   # Dependencies for the project
└── README.md          # Documentation
```
