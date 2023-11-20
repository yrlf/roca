from errno import errorcode
import sys
sys.path.insert(0, './')
import scipy
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from random import seed
import random
from mylib.utils import fix_seed
from mylib.data.data_loader import load_ucidata
import collections
import numpy as np
from run_dnl import run_dnl
import tools
import pandas as pd
from kmeans import run_kmeans
import os
import argparse
import cloth1m_labels
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections
import numpy as np
from fixmatch.utils import *
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from PIL import Image
import torchvision
from torchvision import datasets, transforms
# adjust the load data function to load training data only
from mylib.data.data_loader.load_ucidata import load_ucidata2
from cgi import test
from torch.utils.data.sampler import SubsetRandomSampler
from mylib.data.data_loader.load_cifardata import load_cifardata
from mylib.data.data_loader.load_mnistdata import load_mnistdata
import mylib.data.dataset.util
from mylib.data.dataset.util import get_instance_noisy_label, get_instance_noisy_label2, get_instance_noisy_label3, noisify_multiclass_symmetric


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score
from cal_acc import *
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA



class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """
    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 use_strong_transform=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):

        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets
        
        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform
        self.onehot = onehot
        
        self.transform = transform
        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3,5))
        else:
            self.strong_transform = strong_transform
                
    
    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        
        #set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)
            
        #set augmented images
            
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.use_strong_transform:
                return img_w, target, idx
            else:
                return img_w, self.strong_transform(img), target

    
    def __len__(self):
        return len(self.data)

def _hungarian_match(flat_preds, flat_targets):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]
    v, counts = np.unique(flat_preds, return_counts=True)

    num_k = len(v)
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    row, col = linear_sum_assignment(num_samples - num_correct)
    return row, col


def get_prime_Y(noisy_classes, pred_classes, mappings):
    prime_Y = np.zeros(len(noisy_classes))
    for i in range(len(pred_classes)):
        prime_Y[i] = mappings[pred_classes[i]]

    return prime_Y

# define K-means clustering algorithm

def calculate_acc(ypred, y, return_idx=False):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.

    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.

    """
    assert len(y) > 0
    assert len(np.unique(ypred)) == len(np.unique(y))

    s = np.unique(ypred)
    t = np.unique(y)

    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)

    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    row, col = linear_sum_assignment(C)
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)

    if return_idx:
        return 1.0 * count / len(y), row, col
    else:
        return 1.0 * count / len(y)

def run_kmeans2(dataset):

    X = dataset.data
    clean_Y = dataset.clean_targets
    tilde_Y = dataset.targets
    values, counts = np.unique(clean_Y, return_counts=True)

    kmeans = KMeans(n_clusters=len(values))
    kmeans.fit(X, tilde_Y)
    identified_clusters = kmeans.fit_predict(X)

    # note that to better match the cluster Id to tilde_Y,
    # we could use hat_clean Y which obtained by current noise-robust method, but for the simple dataset , it may not necessary

    idx2 = _hungarian_match(identified_clusters, tilde_Y)
    #idx2 = _hungarian_match(identified_clusters, clean_Y)
    prime_Y = get_prime_Y(tilde_Y, identified_clusters, idx2[1])
    # yz: directly return prime_Y without using count_m
    return prime_Y


def run_kmeans3(dataset, num_classes, feature_size):
    print("************")
    print("kmeans33333")
    print("************")

    X = dataset.data
    # num_classes = dataset._get_num_classes()
    # feature_size = dataset.data.shape[1]
    clean_Y = dataset.clean_targets
    tilde_Y = dataset.targets
    #####
    dataset = dataset.data
    #
    initial_noise_labels = clean_Y
    count = collections.Counter(initial_noise_labels)
    num_classes = len(count)
    train_labels = torch.from_numpy(initial_noise_labels.copy())
    noise_rate = args.init_noise_inject_rate
    norm_std = 0.1
    
    # zip dataset and train_labels
    
    
    dataset2=zip(torch.from_numpy(dataset).float(), torch.from_numpy(initial_noise_labels.copy()))
    tilde_Y, actual_noise_rate, P = get_instance_noisy_label3(n=noise_rate, dataset=dataset2, labels=train_labels, nb_classes=num_classes, feature_size=feature_size, norm_std=norm_std, random_state=args.seed+1)
    
    values, counts = np.unique(clean_Y, return_counts=True)

    kmeans = KMeans(n_clusters=len(values))
    kmeans.fit(X, tilde_Y)
    identified_clusters = kmeans.fit_predict(X)

    # note that to better match the cluster Id to tilde_Y,
    # we could use hat_clean Y which obtained by current noise-robust method, but for the simple dataset , it may not necessary

    idx2 = _hungarian_match(identified_clusters, tilde_Y)
    #idx2 = _hungarian_match(identified_clusters, clean_Y)
    prime_Y = get_prime_Y(tilde_Y, identified_clusters, idx2[1])
    # yz: directly return prime_Y without using count_m
    return prime_Y

########
mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['stl10'] = [0.485, 0.456, 0.406]
mean['npy'] = [0.485, 0.456, 0.406]
mean['npy224'] = [0.485, 0.456, 0.406]
mean['FashionMNIST'] = [0.2860]


std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['stl10'] = [0.229, 0.224, 0.225]
std['npy'] = [0.229, 0.224, 0.225]
std['npy224'] = [0.229, 0.224, 0.225]
std['FashionMNIST'] = [0.3530]

def get_dset(data,targets, num_classes, name, train, use_strong_transform=False, 
                strong_transform=None, onehot=False):
    """
    get_dset returns class BasicDataset, containing the returns of get_data.
    
    Args
        use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
        strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
        onehot: If True, the label is not integer, but one-hot vector.
    """
    
    transform = get_transform(mean[name], std[name], name, train)
    
    return BasicDataset(data, targets, num_classes, transform, 
                        use_strong_transform, strong_transform, onehot)
    

    
def get_transform(mean, std, dataset, train=True):
    if dataset in ['cifar10', 'cifar20', 'cifar100']:
        crop_size = 32
    elif dataset in ['stl10', 'npy']:
        crop_size = 96
    elif dataset in ['FashionMNIST']:
        crop_size = 28
    elif dataset in ['npy224']:
        crop_size = 224
    else:
        raise TypeError
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(crop_size, padding=4),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean, std)])
        
def get_data_loader(dset,
                    batch_size = None,
                    shuffle = False,
                    num_workers = 4,
                    pin_memory = True,
                    data_sampler = None,
                    replacement = True,
                    num_epochs = None,
                    num_iters = None,
                    generator = None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """
    
    assert batch_size is not None
        
    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=num_workers, pin_memory=pin_memory)
    else:
        print("data sample is not implmented")
        

def one_hot(a, num_classes):
    
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def cross_entropy_error(y,t):
    delta=1e-7 
    import torch.nn as nn
    from torch.autograd import Variable
    #y=y.flatten()
    # convert y to one hot encoding
    y=one_hot(y,2)
    y = Variable(torch.FloatTensor(y))
    
    #t = np.expand_dims(t, axis=1)

    t = Variable(torch.LongTensor(t))
    t = Variable(torch.FloatTensor(y.shape[0]).uniform_(0, 1).long())
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y, t)
    return loss.item()
    
    y=y.flatten()
    #print(y.shape[0])
    #print(t.shape[0])
    return -np.sum(y*np.log(t+delta))/y.shape[0]


# --- parsing and configuration --- #
parser = argparse.ArgumentParser(
    description="PyTorch implementation of VAE")
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--z_dim', type=int, default=25,
                    help='dimension of hidden variable Z (default: 10)')
parser.add_argument('--num_hidden_layers', type=int, default=2,
                    help='num hidden_layers (default: 0)')
parser.add_argument('--flip_rate_fixed', type=float,
                    help='fixed flip rates.', default=0.4)
parser.add_argument('--train_frac', default=1.0, type=float,
                    help='training sample size fraction')
parser.add_argument('--noise_type', type=str, default='sym')
parser.add_argument('--trainval_split',  default=0.8, type=float,
                    help='training set ratio')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')
parser.add_argument('--dataset', default="balancescale", type=str,
                    help='db')
parser.add_argument('--select_ratio', default=0, type=float,
                    help='confidence example selection ratio')
parser.add_argument('--pretrained', default=0, type=int,
                    help='using pretrained model or not')
parser.add_argument('--pca_k', type=int, default=5,
                    help='PCA dimension (default: 5)')
parser.add_argument('--sample_size', type=int, default=20,
                    help='randomly select samples for analysis')
parser.add_argument('--near_percentage', type=float, default=0.1,
                    help='percentage nearby in terms of L2 norm')
parser.add_argument('--noise_injection_type',type=str,  default='sym', help='noise injection type, default is symmetric')
parser.add_argument('--init_noise_inject_rate', type=float,help = 'noise injection init', default =0)
'''
Backbone Net Configurations
'''
parser.add_argument('--net', type=str, default='WideResNet')
parser.add_argument('--net_from_name', type=bool, default=False)
parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--widen_factor', type=int, default=2)
parser.add_argument('--leaky_slope', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)

'''
Data Configurations
'''
#parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--data_dir', type=str, default='./datasets/cifar-10-batches-py')
#parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--label_file', type=str, default=None)
parser.add_argument('--all', type=int, default=0)
parser.add_argument('--unlabeled', type=bool, default=False)

parser.add_argument('--output', type =str, default="temp")

arch_dict = {"FashionMNIST": "resnet18", "cifar10": "resnet18", "cifar100": "resnet34", "mnist": "Lenet",
             "balancescale": "NaiveNet", "krkp": "NaiveNet", "splice": "NaiveNet", "yxguassian": "NaiveNet"}

# load dataset

args = parser.parse_args()
if args.dataset == "cifar10" or args.dataset == "cifar10n_worst" or args.dataset == "cifar10n_aggre" or args.dataset == "cifar10n_random1" or args.dataset =='cifar10n_random2' or args.dataset == "cifar10n_random3":
    if args.seed is not None:
        fix_seed(args.seed)
        train_val_loader, train_loader, val_loader, est_loader, test_loader = load_cifardata(
            dataset=args.dataset,
            random_state=args.seed,
            add_noise=True,
            batch_size=args.batch_size, 
            train_frac=args.train_frac, 
            trainval_split=args.trainval_split, 
            noise_type=args.noise_type, 
            flip_rate_fixed=args.flip_rate_fixed, 
            augment=False
        )
else:
    if args.seed is not None:
        fix_seed(args.seed)
        pass
    # train_val_loader, train_loader, val_loader, est_loader, test_loader = load_ucidata(
    #     dataset=args.dataset,
    #     noise_type=args.noise_type,
    #     random_state=args.seed,
    #     batch_size=args.batch_size,
    #     add_noise=True,
    #     flip_rate_fixed=args.flip_rate_fixed,
    #     trainval_split=args.trainval_split,
    #     train_frac=args.train_frac,
    #     augment=False
    # )

# train_dataset = train_loader.dataset
#print("done")
base_dir = "./"+args.dataset+"/"+args.noise_type + \
str(args.flip_rate_fixed)+"/"+str(args.seed)+"/"
#print(args)


def load_dataset_into_list(dataset_list, args,i):
    train_val_loader, train_loader, val_loader, est_loader, test_loader = load_ucidata(
        dataset=args.dataset,
        noise_type=args.noise_type,
        random_state=args.seed+i,
        batch_size=args.batch_size,
        add_noise=True,
        flip_rate_fixed=args.flip_rate_fixed,
        trainval_split=args.trainval_split,
        train_frac=args.train_frac,
        augment=False
    )

    dataset_list.append(train_loader.dataset)


def main():
    #print("************")
    #print(args.noise_type, args.init_noise_inject_rate)
    #print("************")
    # ---- Load data ---- #
    num_classes=-1
    is_balanced=True
    class_dist = []
    primeY =[]
    if args.dataset == "cifar10" or args.dataset =="cifar10n_worst" or args.dataset =="cifar10n_aggre" or args.dataset =="cifar10n_random1" or args.dataset == "cifar10n_random2" or args.dataset =="cifar10n_random3":
        pass
    else:
        # primeY is obtained from K-means unsupervised learning -> we use this as esitmated Clean Y
        if (args.dataset == "cloth1m"):
            pass
        else:
            #print(args.dataset)
            pass
            # primeY = run_kmeans2(train_dataset.dataset)

    # MODIFIED: directly add SYM noise on the noise label
    if (args.dataset == "cifar10"):
        initial_noise_labels = train_dataset.dataset.targets
    elif (args.dataset == "cifar10n_worst"):
        initial_noise_labels = worst_label
    elif (args.dataset == "cifar10n_aggre"):
        initial_noise_labels = aggre_label
    elif (args.dataset == "cifar10n_random1"):
        initial_noise_labels = random1_label
    elif (args.dataset == "cifar10n_random2"):
        initial_noise_labels = random2_label
    elif (args.dataset == "cifar10n_random3"):
        initial_noise_labels = random3_label
    else:
        # initial_noise_labels = train_dataset.dataset.targets
        pass

    # num_classes = train_dataset.dataset._get_num_classes()
    error_list = []

    
    #noisy_rate_list = np.arange(0.1, 0.55, 0.05)
    noisy_rate_list = np.random.uniform(0, 0.5, 10)
    
    norm_std = 0.1
    
    dataset_list = []

    error_list_total = []
    
    
    for noise_rate in noisy_rate_list:
        error_list = []
        
        for i in range(3):
            load_dataset_into_list(dataset_list, args, i)
        for train_dataset in dataset_list:
            num_classes = train_dataset.dataset._get_num_classes()
            feature_size = train_dataset.dataset.data.shape[1]
            if args.noise_type == 'instance' and args.init_noise_inject_rate != 0: # injection noise
                #print("runKmeans3!")
                primeY = run_kmeans3(train_dataset.dataset, num_classes, feature_size)
               
            else:
                primeY = run_kmeans2(train_dataset.dataset)
            dataset2=zip(torch.from_numpy(train_dataset.dataset.data).float(), torch.from_numpy(train_dataset.dataset.targets))
            dataset = train_dataset.dataset.data
            initial_noise_labels = train_dataset.dataset.targets
            if args.noise_injection_type == 'ins':
                #print("UCI, initial noise label shape: ", initial_noise_labels.shape)
                count = collections.Counter(initial_noise_labels)
                num_classes = len(count)
                train_labels = torch.from_numpy(initial_noise_labels.copy())
                # zip dataset and train_labels
                dataset2=zip(torch.from_numpy(dataset).float(), torch.from_numpy(initial_noise_labels.copy()))
                train_noisy_labels, actual_noise_rate, P = get_instance_noisy_label3(n=noise_rate, dataset=dataset2, labels=train_labels, nb_classes=num_classes, feature_size=feature_size, norm_std=norm_std, random_state=args.seed+1)
                count = collections.Counter(train_noisy_labels)
                dist_threshold = 0.05
                is_balanced = True
                num_classes = len(count)
                #print("num_classes: ", num_classes)
                # python max integer
                minority_class = None
                num_minority_instances = sys.maxsize
                for each in count:
                    if count[each] < num_minority_instances:
                        num_minority_instances = count[each]
                        minority_class = each
                    dist = round(count[each]/len(train_noisy_labels), 2)
                    if (dist > 1/num_classes + dist_threshold) or (dist < 1/num_classes - dist_threshold):
                        is_balanced = False

                    class_dist.append((each, dist))

                # subsample the dataset to make it balanced
                index_to_keep = []
                if not is_balanced:
                    primeY = run_kmeans2(train_dataset.dataset)
                    # pick class not equal to minority class
                    for i in range(num_classes-1):
                        if i != minority_class:
                            # randomly select num_minority_instances from class i to keep
                            index_to_keep.extend(random.sample([j for j, x in enumerate(train_noisy_labels) if x == i], num_minority_instances))
                        else:
                            index_to_keep.extend([j for j, x in enumerate(train_noisy_labels) if x == i])
                    
                    train_noisy_labels = train_noisy_labels[index_to_keep]
                    primeY = primeY[index_to_keep]
                    count_resampled = collections.Counter(train_noisy_labels)
            else:
                train_noisy_labels, _, _ = noisify_multiclass_symmetric(initial_noise_labels.copy(
                )[:, np.newaxis], noise_rate, random_state=args.seed+1, nb_classes=num_classes)
            # # calculate error rate
            # res2 = (primeY == train_noisy_labels)
            # count2 = collections.Counter(res2)
            # error_rate = count2[False]/(count2[False]+count2[True])
            # error_list_temp = [error_rate]
            # noise_rate_temp = [noise_rate]
            
            # calculate error rate
            res2 = (primeY == train_noisy_labels)

            # Using NumPy to calculate counts
            count_true = np.sum(res2)
            count_false = res2.size - count_true

            error_rate = count_false / (count_false + count_true)
            error_list_temp = [error_rate]
            noise_rate_temp = [noise_rate]
            
            
            df = pd.concat([pd.DataFrame(noise_rate_temp),
                            pd.DataFrame(error_list_temp)], axis=1)
            df.columns = ['noisy_rate', 'error_rate']
            df['dataset'] = args.dataset
            df['noise_type'] = args.noise_type
            df['causal'] = df['dataset'].apply(lambda x: 1 if x in ["krkp", "balancescale", "splice", "xyguassian"] else 0)
            df['inital_noise'] = args.flip_rate_fixed
            df['inital_noise_ins_injection'] = args.init_noise_inject_rate
            df['seed'] = args.seed
            df['num_classes'] = num_classes
            df['is_balanced'] = is_balanced
            #df['class_dist'] = str(class_dist)
            filename ='./results/results_'+args.output+'.csv'
            # check whether file exists
            if os.path.exists(filename) ==False:
                df.to_csv(filename, header=True, index=False)
            else:
                df.to_csv(filename,
                    mode='a', index=False, header=False)

    #print("all done")


if __name__ == "__main__":
    main()
