MGP-AttTCN: An Interpretable Machine Learning Model for the Prediction of Sepsis
==============================

Data 
------------
The dataset used is the MIMIC III dataset, fount at https://mimic.physionet.org.

Use
------------

STEP I: install dependencies 
`pip install -r requirements.txt`

STEP II: data extraction & preprocessing 
`python scr/data_processing/main.py [-h] -u SQLUSER -pw SQLPASS -ht HOST -db DBNAME -r SCHEMA_READ_NAME [-w SCHEMA_WRITE_NAME]`

STEP III: run the model

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   ├── train          <- The training data used for ... training.
    │   ├── val            <- The validation data used for ... validating (and hyperparameter selection).
    │   └── test           <- The test data used for reporting.
    │
    ├── src                                 <- Source code for use in this project.
    │   ├── __init__.py                     <- Makes src a Python module
    │   │
    │   ├── mains                           <- Runs the full pipeline
    │   │   └── GP_TCN_stat_main.py         <- in use for MGP-TCN; MGP-AttTCN 
    │   │
    │   ├── data_loader                     <- Loads the data into main
    │   │   └── raw_irreg_loader.py         <- in use for MGP-TCN; MGP-AttTCN
    │   │
    │   ├── models                          <- Models to load into main
    │   │   ├── GP_TCN_Moor.py              <- re-implementation of Moor et. al. (MGP-TCN)
    │   │   └── GP_attTCN.py                <- thesis model: MGP + attention based TCN (MGP-AttTCN)
    │   │
    │   ├── trainer                         <- Trains the data
    │   │   └── GP_trainer_with_stat.py     <- in use for MGP-TCN; MGP-AttTCN
    │   │
    │   ├── loss_n_eval                     <- Files to calculate loss, gradients and AUROC, AUPR
    │   │   └── ...
    │   │
    │   ├── visualization                   <- Scripts to create exploratory and results oriented visualizations
    │   │
    │   ├── data_preprocessing              <- Scripts to download or generate data
    │   │
    │   └── features_preprocessing          <- Scripts to turn raw data into features for modeling
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`


Credits 
------------
Credits to M. Moor for sharing his code from https://arxiv.org/abs/1902.01659
