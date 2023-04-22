import click
import torch
import logging
import random
import numpy as np

from utils.visualization.plot_images_grid import plot_images_grid
from datasets.cifar10 import AutoAttck_CIFAR10_Dataset, AutoAttck_CIFAR100_Dataset
from DeepSAD import DeepSAD
from datasets.main import load_dataset
import os
# Get configuration

xp_path = "../log"
data_path = "../data"
dataset_name = 'cifar10'
ratio_known_outlier = 0.5
ratio_known_normal = 0.5
ratio_pollution = 0
n_known_outlier_classes = 9
known_outlier_class = 1
net_name = 'ResNet_18'
load_config = None
num_threads = 0
n_jobs_dataloader = 0
seed = 0
eta = 1.0
ae_optimizer_name = "adam"
ae_lr = 1e-4
ae_n_epochs = 100
ae_lr_milestone = [20, 50, 75]
ae_batch_size = 128
ae_weight_decay = 1e-6
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = xp_path + '/log.txt'
if not os.path.isdir(xp_path):
    os.mkdir(xp_path)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
normal_class = 0

identified_outliers_score = [] 
identified_normals_score = [] 

for target_class in range(0, 10):

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
        logger.info('Network: %s' % net_name)


    # Print model configuration
    logger.info('Eta-parameter: %.2f' % eta)



    # Default device to 'cpu' if cuda is not available
    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
        logger.info('Computation device: %s' % device)
        logger.info('Number of threads: %d' % num_threads)
        logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    # Load data
    dataset = AutoAttck_CIFAR10_Dataset(root=data_path,
                                        normal_class=normal_class,
                                        known_outlier_class=known_outlier_class,
                                        n_known_outlier_classes=n_known_outlier_classes,
                                        ratio_known_normal=ratio_known_normal,
                                        ratio_known_outlier=ratio_known_outlier,
                                        ratio_pollution=ratio_pollution,
                                        target_class = target_class,
                                        advserial_data_path= '../attackDir/aa_standard_50000_Linf_eps_0.03100.pth')

    deepSAD = DeepSAD(eta)
    deepSAD.set_network(net_name)

    deepSAD.load_Resnet_model(model_path=r"../cifar100-ckpt.pth")

    
    deepSAD.train(dataset,
                    optimizer_name=ae_optimizer_name,
                    lr= ae_lr,
                    n_epochs=ae_n_epochs,
                    lr_milestones=ae_lr_milestone,
                    batch_size=ae_batch_size,
                    weight_decay=ae_weight_decay,
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)
    deepSAD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
    deepSAD.save_results(export_json=xp_path + '/results.json')
    deepSAD.save_model(export_model=xp_path + '/model.tar')
    # Plot most anomalous and most normal test samples
    indices, labels, scores = zip(*deepSAD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    identified_outliers = 0
    total_outliers = 0
    identified_normals = 0
    total_normals = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total_outliers += 1
        if labels[i] == 0:
            total_normals += 1
        if labels[i] == 1 and scores[i] > 1:
            identified_outliers += 1
        if labels[i] == 0 and scores[i] < 1:
            identified_normals += 1

    identified_outliers_score.append(identified_outliers/total_outliers)
    identified_normals_score.append(identified_normals/total_normals)

print("identified_outliers_score",identified_outliers_score, sum(identified_outliers_score)/len(identified_outliers_score))
print("identified_normals_score",identified_normals_score, sum(identified_normals_score)/len(identified_normals_score))
