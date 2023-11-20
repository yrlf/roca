import os
import sys
sys.path.append('..')
import numpy as np
from causal_discovery_utils.cond_indep_tests import CondIndepCMI
from causal_discovery_algs import LearnStructRAI, LearnStructPC
from causal_discovery_utils.data_utils import get_var_size
from graphical_models import DAG, PDAG
from causal_discovery_utils.performance_measures import structural_hamming_distance_cpdag, score_bdeu
from experiment_utils.threshold_select_ci_test import search_threshold_bdeu
from matplotlib import pyplot as plt
import random
import numpy as np
from causal_discovery_algs import LearnStructICD, LearnStructFCI
from causal_discovery_utils.cond_indep_tests import CondIndepParCorr  # import a CI test that estimates partial correlation
from experiment_utils.synthetic_graphs import create_random_dag_with_latents, sample_data_from_dag
from causal_discovery_utils.performance_measures import calc_structural_accuracy_pag, find_true_pag
from matplotlib import pyplot as plt
from mylib.data.data_loader import *
import datetime
import csv
import argparse
from random import seed
from mylib.utils import fix_seed
from mylib.data.data_loader import load_ucidata
import numpy as np
import networkx as nx
from cdt.causality.graph import *
from cdt.data import load_dataset
import cdt
import matplotlib.pyplot as plt
import datetime
from stein import *
import networkx as nx
from cdt.causality.graph import PC
from cdt.data import load_dataset
import os
import argparse
from random import seed
from mylib.utils import fix_seed
from mylib.data.data_loader import *
import numpy as np
import csv
import pandas as pd

# --- parsing and configuration --- #
parser = argparse.ArgumentParser(
    description="PyTorch implementation of VAE")
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--z_dim', type=int, default=25,
                    help='dimension of hidden variable Z (default: 10)')
parser.add_argument('--num_hidden_layers', type=int, default=2,
                    help='num hidden_layers (default: 0)')
parser.add_argument('--flip_rate_fixed', type=float,
                    help='fixed flip rates.', default=0.04)
parser.add_argument('--output', type=str,
                    help='output file name', default="run_output.csv")
parser.add_argument('--train_frac', default=1.0, type=float,
                    help='training sample size fraction')
parser.add_argument('--noise_type', type=str, default='pair')
parser.add_argument('--trainval_split',  default=0.8, type=float,
                    help='training set ratio')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')
parser.add_argument('--dataset', default="krkp", type=str,
                    help='db')
parser.add_argument('--select_ratio', default=0, type=float,
                    help='confidence example selection ratio')
parser.add_argument('--pretrained', default=0, type=int,
                    help='using pretrained model or not')

parser.add_argument('--method', type=str, default="fci",
                    help='number of epochs to train (default: 20)')


def runCI(data_train, graph_nodes):
    CITest = CondIndepCMI
    th_range = [i / 10000 + 0.01 for i in range(100)]
    th_pc, all_scores_pc = search_threshold_bdeu(LearnStructPC, data_train, CITest, th_range)
    print('Selected PC threshold = {:.4f}'.format(th_pc))
    ci_test_pc = CITest(dataset=data_train, threshold=th_pc, count_tests=True)  # conditional independence test
    pc = LearnStructPC(nodes_set=graph_nodes, ci_test=ci_test_pc)
    pc.learn_structure()  # learn structure
    return pc


def runRAI(data_train, graph_nodes):
    CITest = CondIndepCMI
    th_range = [i / 10000 + 0.01 for i in range(100)]
    th_rai, all_scores_rai = search_threshold_bdeu(LearnStructRAI, data_train, CITest, th_range)
    print('Selected RAI threshold = {:.4f}'.format(th_rai))
    ci_test_rai = CITest(dataset=data_train, threshold=th_rai, count_tests=True)
    rai = LearnStructRAI(nodes_set=graph_nodes, ci_test=ci_test_rai)
    rai.learn_structure()  # learn structure
    return rai

def runBRAI(data_train, graph_nodes):
    pass


def runICD(data_train, graph_nodes):
    alpha = 0.01
    par_corr_icd = CondIndepParCorr(
        dataset=data_train,
        threshold=alpha,
        count_tests=True,
        use_cache=True
    )
    icd = LearnStructICD(graph_nodes, par_corr_icd)  # instantiate an ICD learner
    icd.learn_structure()  # learn the PAG
    learned_pag_icd = icd.graph
    return icd


def runFCI(data_train, graph_nodes):
    alpha = 0.01
    par_corr_fci = CondIndepParCorr(dataset=data_train, threshold=alpha, count_tests=True, use_cache=True)  # CI test
    fci = LearnStructFCI(graph_nodes, par_corr_fci)  # instantiate an ICD learner
    fci.learn_structure()  # learn the PAG

    return fci


def choooseMethod(method, data_train, graph_nodes):
    if method == 'fci':
        obj = runFCI(data_train, graph_nodes)
    elif method == 'icd':
        obj = runICD(data_train, graph_nodes)
    elif method == 'rai':
        obj = runRAI(data_train, graph_nodes)
    elif method == 'ci':
        obj = runCI(data_train, graph_nodes)
    else:
        print("unknown method choosen")
        obj = None

    return obj


def checkCA(model, dataset):
    graph = model.graph
    results = "unkown"
    try:
        dag = DAG(graph.nodes_set)
        graph.convert_to_dag(dag)
        print(dag.get_adj_mat()[0])
        outedge = dag.get_adj_mat()[0][-1].sum()
        inedge = dag.get_adj_mat()[0].sum(axis=0)[-1]
    except:
        print(graph.get_adj_mat())
        outedge = graph.get_adj_mat()[-1].sum()
        inedge = graph.get_adj_mat().sum(axis=0)[-1]

    if outedge > 0:
        results = "anticausal"
        print(dataset, "anticausal", outedge)

    elif inedge > 0:
        results = "causal"
        print(dataset, "causal", inedge)

    else:
        print(dataset, "unkown")

    return results

    # from plot_utils import draw_graph
    # fig = draw_graph(dag)


def draws(model):
    dag = DAG(model.graph.nodes_set)
    try:
        dag = DAG()
        graph.convert_to_dag(dag)
        outedge = dag.get_adj_mat()[0][-1].sum()
        inedge = dag.get_adj_mat()[0].sum(axis=0)[-1]
    except:
        outedge = graph.get_adj_mat()[-1].sum()
        inedge = graph.get_adj_mat().sum(axis=0)[-1]

    model.graph.convert_to_dag(dag)
    from plot_utils import draw_graph
    fig = draw_graph(dag)
    fig.show()

############## aaa.py ########
def generate(d, s0, N, noise_std=1, noise_type='Gauss', graph_type='ER', GP=True, lengthscale=1):
    adjacency = simulate_dag(d, s0, graph_type, triu=True)
    teacher = Dist(d, noise_std, noise_type, adjacency, GP=GP, lengthscale=lengthscale)
    X, noise_var = teacher.sample(N)
    return X, adjacency


# Data generation paramters
graph_type = 'ER'
d = 10
s0 = 10
N = 1000

X2, adj = generate(d, s0, N, GP=True)


def checkCA2(dataset, arr):
    outedge = np.sum(arr[-1] == 1)
    inedge = np.sum(arr[:, -1] == 1)
    print("out edge: " + str(outedge) + " in edge: " + str(inedge))

    results = "unkown"
    if outedge > 0:
        results = "anticausal"
        print(dataset, "anticausal", outedge)

    elif inedge > 0:
        results = "causal"
        print(dataset, "causal", inedge)

    else:
        print(dataset, "unkown")

    return results


# data, graph = load_dataset("sachs")

def runCDT(method):
    if method == 'pc':
        obj = PC()
    elif method == 'ges':
        obj = GES()
    elif method == 'lingam':
        obj = LiNGAM()
    elif method == 'cam':
        obj = CAM(pruning=False, selmethod='lasso', variablesel=False, verbose=True)
    elif method == 'cgnn':
        obj = CGNN(nh=5, train_epochs=100, test_epochs=100, gpus=1, verbose=True, nruns=2)
    elif method == 'gies':
        obj = GIES()
    elif method == 'sam':
        obj = SAM(train_epochs=300, test_epochs=100)
        print("SAM object created")
    elif method == 'ccdr':
        obj = CCDr()
    else:
        print("unknown method")
        obj = None
    return obj


def runSCORE(X):
    # SCORE hyper-parameters
    eta_G = 0.001
    eta_H = 0.001
    cam_cutoff = 0.001

    A_SCORE, top_order_SCORE = SCORE(X, eta_G, eta_H, cam_cutoff, pruning='Stein')
    return A_SCORE
def main():

    args = parser.parse_args()
    base_dir = "./"+args.dataset+"/"+args.noise_type+str(args.flip_rate_fixed)+"/"+str(args.seed)+"/"
    print(args)

    if args.seed is not None:
        fix_seed(args.seed)

    if args.dataset =="cifar10":
        train_val_loader, train_loader, val_loader, est_loader, test_loader = load_cifardata(
            dataset=args.dataset,
            noise_type=args.noise_type,
            random_state=args.seed,
            batch_size=args.batch_size,
            add_noise=True,
            flip_rate_fixed=args.flip_rate_fixed,
            trainval_split=args.trainval_split,
            train_frac=args.train_frac,
            augment=False
        )

    else:
        train_val_loader, train_loader, val_loader, est_loader, test_loader = load_ucidata(
            dataset = args.dataset,
            noise_type = args.noise_type,
            random_state = args.seed,
            batch_size = args.batch_size,
            add_noise = True,
            flip_rate_fixed = args.flip_rate_fixed,
            trainval_split = args.trainval_split,
            train_frac = args.train_frac,
            augment=False
        )

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    val_dataset = val_loader.dataset

    if args.dataset == 'cifar10':
        data_train = train_dataset.dataset.data
        # reshape to 2d
        data_train = data_train.reshape(data_train.shape[0], -1)
        noisy_labels = train_dataset.dataset.targets
        noisy_labels = np.reshape(noisy_labels, (-1, 1))
        data_train = np.concatenate((data_train, noisy_labels), axis=1)

        print("data_train shape is: ", data_train.shape)
    else:  # UCI dataset
        data_train = train_loader.dataset.dataset.noise_data

    n_samples, n_vars = data_train.shape  # data is assumed a numpy 2d-array
    graph_nodes = set(range(n_vars))  # create a set containing the nodes indices
    print(n_samples, n_vars)

    start_time = datetime.datetime.now()
    if args.method in ['pc', 'ges', 'lingam', 'cam', 'cgnn', 'gies', 'sam', 'ccdr','score']: #aaa.py
        data = pd.DataFrame(data_train, columns=list(range(n_vars)))
        if args.method == 'score':
            if (args.dataset == 'cifar10'):
                X = torch.tensor(data_train)

            else:
                data.iloc[:, -1] = pd.Categorical(data.iloc[:, -1])
                X = torch.tensor(data.values)
            a_score = runSCORE(X)
            print(a_score)
            results = checkCA2(args.dataset, a_score)
        else:
            print("runCDT starting")
            obj = runCDT(args.method)
            print("obj predict starting:")
            output = obj.predict(data)  # No graph provided as an argument
            print("obj predict finished, starting to plot adjacency matrix")
            output_graph = nx.adjacency_matrix(output)
            matrix_dense = output_graph.toarray()
            results = checkCA2(args.dataset, matrix_dense)
    else: #run.py

        obj = choooseMethod(args.method, data_train, graph_nodes)

        # rai = runICD(data_train,graph_nodes)
        results = checkCA(obj, args.dataset)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    data_to_save = [args.dataset, args.noise_type, str(args.flip_rate_fixed), args.method, str(results), str(elapsed_time)]

    headers = ['Dataset', 'Noise_Type', 'Noise_Rate', 'Method', "Causality", "Time"]
    filename = args.output


    if os.path.isfile(filename):
        with open(filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data_to_save)
    else:
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(data_to_save)

if __name__ == "__main__":
    main()
