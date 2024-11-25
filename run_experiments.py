from train import process
import models
from subprocess import call
import argparse
import json
import copy

def run(params, gpu):
    json.dump(params, open("params_gpu%d.json" % gpu, "w"))
    call("python train.py -p params_gpu%d.json" % gpu)

def add_extra_params(params, extra_params):
    if extra_params != '':
        new_params = extra_params.replace("'", "").split("&")
        for p in new_params:
            t, v = p.split("=")
            t1, t2 = t.split(".")
            try:
                params[t1][t2] = eval(v)
            except:
                params[t1][t2] = v
            print(t1, t2, params[t1][t2])

def get_configuration(suffix, meta_suffix='bow', meta_suffix2='bow', meta_suffix3='bow', meta_suffix4='bow', extra_params=''):
    params = dict()

    # LOGISTIC multimodal 3
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel'  # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'class'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 250  # 397
    nparams["training"]["loss_func"] = 'binary_crossentropy'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = False
    nparams["cnn"]["architecture"] = '33'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.0
    nparams["cnn"]["final_activation"] = 'sigmoid'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-albums'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["dataset"]["meta-suffix2"] = meta_suffix2
    nparams["dataset"]["meta-suffix3"] = meta_suffix3
    add_extra_params(nparams, extra_params)
    params['logistic_multilabel_tri'] = copy.deepcopy(nparams)

    # COSINE multimodal 3
    nparams = copy.deepcopy(models.params_1)
    nparams["dataset"]["evaluation"] = 'multilabel'  # binary/multilabel/multiclass/recommendation
    nparams["dataset"]["fact"] = 'pmi'
    nparams["dataset"]["npatches"] = 1
    nparams["dataset"]["with_metadata"] = True
    nparams["dataset"]["only_metadata"] = True
    nparams["dataset"]["configuration"] = suffix
    nparams["dataset"]["sparse"] = False
    nparams["training"]["val_from_file"] = True
    nparams["dataset"]["dim"] = 50
    nparams["training"]["loss_func"] = 'cosine'
    nparams["training"]["optimizer"] = 'adam'
    nparams["training"]["normalize_y"] = True
    nparams["cnn"]["architecture"] = '33'
    nparams["cnn"]["n_dense"] = 0
    nparams["cnn"]["dropout_factor"] = 0.7
    nparams["cnn"]["final_activation"] = 'linear'
    nparams["dataset"]["nsamples"] = 'all'
    nparams["dataset"]["dataset"] = 'MuMu-albums'
    nparams["dataset"]["meta-suffix"] = meta_suffix
    nparams["dataset"]["meta-suffix2"] = meta_suffix2
    nparams["dataset"]["meta-suffix3"] = meta_suffix3
    add_extra_params(nparams, extra_params)
    params['cosine_multilabel_tri'] = copy.deepcopy(nparams)

    return params[suffix]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('suffix', default="class_bow", help='Suffix of experiment params')
    parser.add_argument('meta_suffix', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('meta_suffix2', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('meta_suffix3', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('meta_suffix4', nargs='?', default="", help='Suffix of input matrix for experiment')
    parser.add_argument('extra_params', nargs='?', default="", help='Specific extra parameters')
    args = parser.parse_args()
    print(args.extra_params)
    params = get_configuration(args.suffix, args.meta_suffix, args.meta_suffix2, args.meta_suffix3, args.meta_suffix4, args.extra_params)
    process(params)