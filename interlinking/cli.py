import click
import os
from interlinking import pre_process
from interlinking.learning import learn_thres, learn_params_for_lgm
from interlinking import core


@click.group()
def cli():
    pass


@cli.command('extract_frequent_terms', help='Create a file with ranked frequent terms found in corpus')
@click.option('--train_set', default='dataset-string-similarity_global_1k.csv',
              help='Train dataset to extract frequent terms from. '
                   'It requires once to run: python -m nltk.downloader \'punkt\'')
@click.option('--encoding', default='latin', type=click.Choice(['latin', 'global']),
              help='specify the alphabet encoding of toponyms in dataset.')
def freq_terms(train_set, encoding):
    pre_process.extract_freqterms(train_set, encoding)


@cli.command('learn_sim_params', help='Learn parameters, i.e., weights/thresholds, on a train dataset for '
                                      'similarity metrics')
@click.option('--train_set', default='dataset-string-similarity_global_1k.csv',
              help='Train dataset to learn parameters.')
@click.option('--sim_type', default='basic', type=click.Choice(['basic', 'sorted', 'lgm']),
              help='Group of similarities to train. Valid options are: basic, sorted, lgm')
@click.option('--encoding', default='latin', type=click.Choice(['latin', 'global']),
              help='specify the alphabet encoding of toponyms in dataset.')
def learn_params(train_set, sim_type, encoding):
    if sim_type == 'lgm':
        learn_params_for_lgm(train_set, encoding)
    else: learn_thres(train_set, sim_type)


@cli.command('hyperparameter_tuning', help='Tune various classifiers and select the best hyper-parameters on a '
                                           'train dataset')
@click.option('--train_set', default='dataset-string-similarity_global_1k.csv',
              help='Train dataset to learn parameters.')
@click.option('--test_set', default='dataset-string-similarity.txt', help='Test dataset to evaluate the models.')
@click.option('--encoding', default='latin', type=click.Choice(['latin', 'global']),
              help='specify the alphabet encoding of toponyms in dataset.')
def hyperparams_learn(train_set, test_set, encoding):
    core.StrategyEvaluator(encoding).hyperparamTuning(train_set, test_set)


@cli.command('evaluate', help='Tune various classifiers and select the best hyper-parameters on a train dataset')
@click.option('--train_set', default='', help='Train dataset to learn parameters.')
@click.option('--test_set', default='dataset-string-similarity.txt', help='Test dataset to validate the models.')
@click.option('--encoding', default='latin', type=click.Choice(['latin', 'global']),
              help='specify the encoding of toponyms in dataset.')
def eval_classifiers(train_set, test_set, encoding):
    core.StrategyEvaluator(encoding).evaluate(train_set, test_set)


if __name__ == '__main__':
    cli()
