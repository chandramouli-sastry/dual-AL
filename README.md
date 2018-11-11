# dual-AL

The code is written in Python 2.7 and requires the following primary packages.
```
Tensorflow == 1.7
PyTorch == 0.4
Numpy
NLTK
Gensim == 3.3.0
```

The required data and embeddings are available here: https://www.dropbox.com/s/hu74dagau77kcq5/Data.zip?dl=0
(Extract and place all the folders in the root level)

In order to run text classification:
```
python -m classification.driver [pangimdb|aclimdb] [cnn|rnn] [0|1]
python -m classification.driver aclimdb_sharma
```

In order to run classification with feature expert:
```
python -m featExpert_AL.driver [cnn|rnn] [0|1] [nova|sraa|WvsH]
```

In order to run information retrieval experiments:
```
python -m IR.driver [cnn|rnn] [0|1] [Hall|Kitchenham|Wahano|Radjenovic]
```

In order to run one shot classification experiments:
```
python -m 1shot.driver [cnn|rnn|sharma]
```

Some more commenting to be done! 
