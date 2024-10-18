# Alg-MFDL Framework



## Introduction

We proposed a novel deep learning model, named Alg-MFDL, to accurately identify allergenic proteins by integrating pre-trained PLMs and traditional handcrafted features. This predictor utilizes three advanced deep learning networks to effectively train and refine its predictive abilities, including CNN, transformer, and bidirectional gated recurrent units (BiGRUs). 
The model achieved an ACC of 0.973, a precision of 0.996, a SN of 0.951, and a F1 score of 0.973 on the independent set.



## Requirements

```
numpy==1.23.5
pandas==1.4.4
matplotlib==3.7.1
scikit_learn==0.24.2
torch==1.13.1+cu117
torchvision==0.14.1+cu117

```

## Usage



run `get_Features/get_AAC-PSSM_feature.py` to generate AAC-PSSM feature files. 

run `get_Features/get_DPC-PSSM_feature.py` to generate DPC-PSSM feature files. 

run `get_Features/get_DDE.py` to generate DDE feature files. 

run `get_Features/get_ESM2.py` to generate ESM2 feature files.

run `get_Features/get_Prott5.py` to generate Prott5 feature files.

run `get_Features/get_ProtXLNet.py` to generate ProtXLNet feature files.

run `get_Features/get_Protbert.py` to generate Protbert feature files.

run `get_Features/get_ProtAlbert.py` to generate ProtAlbert feature files.

run `Model_train/Train.py` and `Model_train/Test.py` to obtain the results of the 5-fold CV and independent test set of the model.

## Acknowledge


The training and test sets come from DeepAlgPro (DOI: https://doi.org/10.1093/bib/bbad246) .
