|MIT|

=====

================
LGM-Interlinking
================
This Python code implements and evaluates the proposed LinkGeoML models for Toponym classification-based interlinking.

In this setting, we consider the names of the toponyms as the only source of information that can be used to decide
whether two toponyms refer to the same real-world entity.
Specifically, we build a meta-similarity function, called *LGM-Sim*, that takes into account
and incorporates within its processing steps the specificities of toponym names. Consequently, we derive training
features from **LGM-Sim** that are used in various classification models. The proposed method and its derived features
are robust enough to handle variations in the distribution of toponyms and demonstrate a significant increase in
interlinking accuracy compared to baseline models widely used in the literature (see `References`_). Indicatively, we
succeed a 85.6% accuracy with the Gradient Boosting Trees classifier compared to the best baseline model that achieves
accuracy of 78.6% with Random Forests.

The *data* folder contains the train datasets, which are used to build the classifiers, and files containing frequent
terms, extracted from train datasets. For evaluation, we used the dataset from
the `Toponym-Matching <https://github.com/ruipds/Toponym-Matching>`_ work (see `Setup procedure`_).

..
    The **scripts** folder contains the evaluation setting used to execute the experiments and collect the results presented in the paper:
      - `./scripts/basic_train_latin.sh`: collect the effectiveness values for the **basic** setup on the **100k latin** dataset;
      - `./scripts/lgm_train_latin.sh`: collect the effectiveness values for the **LGM** setup on the **100k latin** dataset;
      - `./scripts/basic_train_global.sh`: collect the effectiveness values for the **basic** setup on the **100k global** dataset;
      - `./scripts/lgm_train_global.sh`: collect the effectiveness values for the **LGM** setup on the **100k global** dataset;
      - `./scripts/basic_test_100klatin_parameter_based.sh`: collect the effectiveness values for the **basic** setup on the global dataset with hyper parameters obtained on the **100k latin train** dataset;
      - `./scripts/lgm_test_100klatin_parameter_based.sh`: collect the effectiveness values for the LGM setup on the global dataset with hyper parameters obtained on the **100k latin train** dataset;
      - `./scripts/basic_test_100kglobal_parameter_based.sh`: collect the effectiveness values for the **basic** setup on the global dataset with hyper parameters obtained on the **100k global train** dataset;
      - `./scripts/lgm_test_100kglobal_parameter_based.sh`: collect the effectiveness values for the **LGM** setup on the global dataset with hyper parameters obtained on the **100k global train** dataset.

The source code was tested using Python 3 (>=3.6) and Scikit-Learn 0.22.2.post1 on a Linux server.

Setup procedure
---------------
Download the latest version from the `GitHub repository <https://github.com/LinkGeoML/LGM-Interlinking.git>`_, change to
the main directory and run:

.. code-block:: bash

   pip install -r pip_requirements.txt

It should install all the required libraries automatically (*scikit-learn, numpy, pandas etc.*).

Change to the **data** folder, download the test dataset and unzip it:

.. code-block:: bash

   wget https://github.com/ruipds/Toponym-Matching/raw/master/dataset/dataset.zip
   wget https://github.com/ruipds/Toponym-Matching/raw/master/dataset/dataset.z01

   zip -FF dataset.zip  --out dataset.zip.fixed
   unzip dataset.zip.fixed


How to use
----------
The input dataset need to be in CSV format. Specifically, a valid dataset should have at least the following
fields/columns:

* The names for each of the candidate toponym pairs.
* The label, i.e., {True, False}, assigned to each toponym pair.

The library implements the following distinct processes:

#. Features extraction
    The `build <https://linkgeoml.github.io/LGM-Interlinking/features.html#interlinking.features.Features>`_ function
    constructs a set of training features to use within classifiers for toponym interlinking.

#. Algorithm and model selection
    The functionality of the
    `fineTuneClassifiers <https://linkgeoml.github.io/LGM-Interlinking/tuning.html#interlinking.hyperparam_tuning.
    ParamTuning.fineTuneClassifiers>`_ function is twofold.
    Firstly, it chooses among a list of supported machine learning algorithms the one that achieves the highest average
    accuracy score on the examined dataset. Secondly, it searches for the best model, i.e., the best hyper-parameters
    for the best identified algorithm in the first step.

#. Model training
    The `trainClassifier <https://linkgeoml.github.io/LGM-Interlinking/tuning.html#interlinking.hyperparam_tuning.
    ParamTuning.trainClassifier>`_ trains the best selected model on previous
    process, i.e., an ML algorithm with tuned hyperparameters that best fits data, on the whole train dataset, without
    splitting it in folds.

#. Model deployment
    The `testClassifier <https://linkgeoml.github.io/LGM-Interlinking/tuning.html#interlinking.hyperparam_tuning.
    ParamTuning.testClassifier>`_ applies the trained model on new untested data.

A complete pipeline of the above processes, i.e., features extraction, training and evaluating state-of-the-art
classifiers, for toponym interlinking can be executed with the following command:

.. code-block:: bash

    $ python -m interlinking.cli hyperparameter_tuning --train_set <path/to/train-dataset>
    --test_set <path/to/test-dataset>

Additionally, *help* is available on the command line interface (*CLI*). Enter the following to list all supported
commands or options for a given command with a short description.

.. code-block:: bash

    $ python -m interlinking.cli -h
    Usage: cli.py [OPTIONS] COMMAND [ARGS]...

    Options:
      -h, --help  Show this message and exit.

    Commands:
      build                   build a candidate pairs of toponyms dataset for evaluation from Geonames
      evaluate                evaluate the effectiveness of the proposed methods
      extract_frequent_terms  create a file with ranked frequent terms found in corpus
      hyperparameter_tuning   tune various classifiers and select the best hyper-parameters on a train dataset
      learn_sim_params        learn parameters, i.e., weights/thresholds, on a train dataset for similarity metrics

Documentation
-------------
Source code documentation is available from `linkgeoml.github.io`__.

__ https://linkgeoml.github.io/LGM-Interlinking/


Acknowledgements
-------------------
The *sim_measures.py* file, which is used to generate the train/test datasets and to compute the string similarity
measures, is a slightly modified version of the *datasetcreator.py* file used in
`Toponym-Matching <https://github.com/ruipds/Toponym-Matching>`_ work, which is under the MIT license.

References
----------
* Santos, R., Murrieta-Flores, P. and Martins, B., 2018. Learning to combine multiple string similarity metrics for
  effective toponym matching. *International journal of digital earth*, 11(9), pp.913-938.


License
-------
LGM-Interlinking is available under the `MIT <https://opensource.org/licenses/MIT>`_ License.

..
    .. |Documentation Status| image:: https://readthedocs.org/projects/coala/badge/?version=latest
       :target: https://linkgeoml.github.io/LGM-Interlinking/

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
