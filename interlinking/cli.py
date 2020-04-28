# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@athenarc.gr

import click

from interlinking import pre_process
from interlinking.learning import learn_thres, learn_params_for_lgm
from interlinking.sim_measures import build_dataset
from interlinking import core


@click.group(context_settings=dict(max_content_width=120, help_option_names=['-h', '--help']))
def cli():
    pass


@cli.command('build', help='\b build a candidate pairs of toponyms dataset for evaluation from Geonames')
@click.option('--dataset', default='allCountries.txt', help='.')
@click.option('--n_alternates', default='3', help='Min number of alternative names in order to process the record.')
@click.option('--num_instances', default='5000000',
              help='total number of toponym pairs to create per flag (true/false).')
@click.option('--encoding', default='latin', type=click.Choice(['latin', 'global']),
              help='specify the alphabet encoding of toponyms in dataset.')
def build(dataset, n_alternates, num_instances, encoding):
    build_dataset(dataset, n_alternates, num_instances, encoding)


@cli.command('extract_frequent_terms', help='create a file with ranked frequent terms found in corpus')
@click.option('--train_set', default='dataset-string-similarity_global_1k.csv',
              help='Train dataset to extract frequent terms from. '
                   'It requires once to run: python -m nltk.downloader \'punkt\'')
@click.option('--encoding', default='latin', type=click.Choice(['latin', 'global']),
              help='specify the alphabet encoding of toponyms in dataset.')
def freq_terms(train_set, encoding):
    pre_process.extract_freqterms(train_set, encoding)


@cli.command('learn_sim_params', help='learn parameters, i.e., weights/thresholds, on a train dataset for '
                                      'similarity metrics')
@click.option('--train_set', default='dataset-string-similarity_global_1k.csv',
              help='Train dataset to learn parameters.')
@click.option('--sim_type', default='basic', type=click.Choice(['basic', 'sorted', 'lgm']),
              help='Group of similarities to train. Valid options are: basic, sorted, lgm')
@click.option('--encoding', default='latin', type=click.Choice(['latin', 'global']),
              help='Specify the alphabet encoding of toponyms in dataset.')
def learn_params(train_set, sim_type, encoding):
    if sim_type == 'lgm':
        learn_params_for_lgm(train_set, encoding)
    else: learn_thres(train_set, sim_type)


@cli.command('hyperparameter_tuning', help='tune various classifiers and select the best hyper-parameters on a '
                                           'train dataset')
@click.option('--train_set', default='dataset-string-similarity_global_1k.csv',
              help='Train dataset to learn parameters.')
@click.option('--test_set', default='dataset-string-similarity.txt', help='Test dataset to evaluate the models.')
@click.option('--encoding', default='latin', type=click.Choice(['latin', 'global']),
              help='Specify the alphabet encoding of toponyms in dataset.')
def hyperparams_learn(train_set, test_set, encoding):
    core.StrategyEvaluator(encoding).hyperparamTuning(train_set, test_set)


@cli.command('evaluate', help='')
@click.option('--train_set', default='', help='the dataset to train the models.')
@click.option('--test_set', default='dataset-string-similarity.txt', help='the dataset to apply/evaluate trained models.')
@click.option('--encoding', default='latin', type=click.Choice(['latin', 'global']),
              help='Specify the encoding of toponyms in dataset.')
def eval_classifiers(train_set, test_set, encoding):
    core.StrategyEvaluator(encoding).evaluate(train_set, test_set)


if __name__ == '__main__':
    cli()
