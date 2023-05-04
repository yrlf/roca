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
# --- parsing and configuration --- #
parser = argparse.ArgumentParser(
    description="PyTorch implementation of VAE")
parser.add_argument('--output', type=str, default="aaa_results.csv",
                    help='output filename')

parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--z_dim', type=int, default=25,
                    help='dimension of hidden variable Z (default: 10)')
parser.add_argument('--num_hidden_layers', type=int, default=2,
                    help='num hidden_layers (default: 0)')
parser.add_argument('--flip_rate_fixed', type=float,
                    help='fixed flip rates.', default=0.02)
parser.add_argument('--train_frac', default=1.0, type=float,
                    help='training sample size fraction')
parser.add_argument('--noise_type', type=str, default='pair')
parser.add_argument('--trainval_split',  default=0.8, type=float,
                    help='training set ratio')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')
parser.add_argument('--dataset', default="iris", type=str,
                    help='db')
parser.add_argument('--select_ratio', default=0, type=float,
                    help='confidence example selection ratio')
parser.add_argument('--pretrained', default=0, type=int,
                    help='using pretrained model or not')

parser.add_argument('--method', type=str, default="pc",
                    help='number of epochs to train (default: 20)')





def generate(d, s0, N, noise_std = 1, noise_type = 'Gauss', graph_type = 'ER', GP = True, lengthscale=1):
    adjacency = simulate_dag(d, s0, graph_type, triu=True)
    teacher = Dist(d, noise_std, noise_type, adjacency, GP = GP, lengthscale=lengthscale)
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
    inedge =  np.sum(arr[:, -1] == 1)
    print("out edge: "+str(outedge)+" in edge: "+str(inedge))

    results = "unkown"
    if outedge>0:
        results = "anticausal"
        print(dataset, "anticausal",outedge)

    elif inedge>0:
        results = "causal"
        print(dataset, "causal", inedge)

    else: 
        print(dataset, "unkown")

    return results
    
# data, graph = load_dataset("sachs")

def runCDT(method):
    if method=='pc':
        obj = PC()
    elif method =='ges':
        obj = GES()
    elif method =='lingam':
        obj=LiNGAM()
    elif method =='cam':
        obj = CAM(pruning=False, selmethod='lasso', variablesel=False, verbose=True)
    elif method =='cgnn':
        obj =CGNN(nh=5, train_epochs=100, test_epochs=100,gpus=1,  verbose=True, nruns=2)
    elif method =='gies':
        obj =GIES()
    elif method =='sam':
        obj =SAM(train_epochs=300, test_epochs=100)
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
    
    A_SCORE, top_order_SCORE =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning='Stein')
    return A_SCORE

args = parser.parse_args()
base_dir = "./"+args.dataset+"/"+args.noise_type+str(args.flip_rate_fixed)+"/"+str(args.seed)+"/"
print(args)

if args.seed is not None:
    fix_seed(args.seed)

if args.dataset == 'cifar10':
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
    # combine data and labels
    # print("shape of data_train is: ", data_train.shape)
    # print("shape of noisy_labels is: ", noisy_labels.shape)

    noisy_labels = np.reshape(noisy_labels, (-1, 1))
    data_train = np.concatenate((data_train, noisy_labels), axis=1)
    print("data_train shape is: ", data_train.shape)
    #print(data_train)

else:
    data_train = train_loader.dataset.dataset.noise_data
# print(data_train)

n_samples, n_vars = data_train.shape

print(n_vars-1)
import pandas as pd
data = pd.DataFrame(data_train, columns = list(range(n_vars)))


# print(df)

start_time = datetime.datetime.now()
if args.method == 'score':
    if (args.dataset == 'cifar10'):
        X=torch.tensor(data_train)

    else:
        data.iloc[:,-1] = pd.Categorical(data.iloc[:,-1])
        X=torch.tensor(data.values)
    a_score= runSCORE(X)
    print(a_score)
    results = checkCA2(args.dataset, a_score)
#if (args.method == 'pc' or args.method == 'ges' or args.method == 'cam' or args.method == 'lingam' or args.method == 'gies'):
else:
    print("runCDT starting")
    obj = runCDT(args.method)
    # print(data)
    #The predict() method works without a graph, or with a
    #directed or undirected graph provided as an input
    print("obj predict starting:")
    output = obj.predict(data)    # No graph provided as an argument
    #exit()
    #checkCA(output)
    # output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
    # output = obj.predict(data, graph)  #With a directed graph
    # print(output)
    #To view the graph created, run the below commands:
    print("obj predict finished, starting to plot adjacency matrix")
    output_graph= nx.adjacency_matrix(output)

    matrix_dense=output_graph.toarray()

    results = checkCA2(args.dataset,matrix_dense)

end_time = datetime.datetime.now()
elapsed_time = end_time - start_time
print("Time elapsed: ", elapsed_time)
data_to_save = [args.dataset, args.noise_type, str(args.flip_rate_fixed), args.method, str(results), str(elapsed_time)]

headers = ['Dataset', 'Noise_Type', 'Noise_Rate','Method',"Causality","Time"]

filename=args.output

if os.path.isfile(filename):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        
        writer.writerow(data_to_save)
else:
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        writer.writerow(data_to_save)

#nx.draw_networkx(output, font_size=8)
#plt.show()


# data, graph = cdt.data.load_dataset('sachs')
# glasso = cdt.independence.graph.Glasso()
# skeleton = glasso.predict(data)
# # new_skeleton = nx.DiGraph()

# model = LiNGAM()
# output_graph = model.predict(data,skeleton)
# print(nx.adjacency_matrix(output_graph).todense())


# import networkx as nx
# g = nx.DiGraph()  # initialize a directed graph
# l = list(g.nodes())  # list of nodes in the graph
# a = nx.adj_matrix(g).todense()  # Output the adjacency matrix of the graph
# e = list(g.edges())

# data, graph = load_dataset("sachs")
# obj = LiNGAM()
# output = obj.predict(data)
