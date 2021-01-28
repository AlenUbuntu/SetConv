from os import sys, path

from config import cfg 
from utils import setup_logger
from torch.utils.collect_env import get_pretty_env_info
from data import make_data_loader, load_dataset
from model import build_model
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score, specificity_score, sensitivity_score
from sklearn.metrics import balanced_accuracy_score
import argparse 
import os 
import torch 
import numpy as np 
import tqdm



def make_optimizer(cfg, model):
    if cfg.OPTIMIZER.NAME == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.downsample.parameters(), 'lr': cfg.OPTIMIZER.FE_LR},
            {'params': model.set_conv.parameters(), 'lr': cfg.OPTIMIZER.FE_LR},
            {'params': model.fc.parameters(), 'lr': cfg.OPTIMIZER.BASE_LR}
        ], betas=cfg.OPTIMIZER.BETAS, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    elif cfg.OPTIMIZER.NAME == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': model.downsample.parameters(), 'lr': cfg.OPTIMIZER.FE_LR},
            {'params': model.set_conv.parameters(), 'lr': cfg.OPTIMIZER.FE_LR},
            {'params': model.fc.parameters(), 'lr': cfg.OPTIMIZER.BASE_LR}
        ], momentum=cfg.OPTIMIZER.MOMENTUM, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise NotImplementedError("{} is not supported yet".format(cfg.OPTIMIZER.NAME))
    
    return optimizer

def make_lr_scheduler(cfg, optimizer):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.OPTIMIZER.MAX_EPOCH * cfg.DATALOADER.NUM_BATCH
    )
    return lr_scheduler

def train(cfg, model, train_loader, valid_loader, save=False):
    best_epoch = ''
    best_score = 0

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # create optimizer
    optimizer = make_optimizer(cfg, model)

    # create learning rate scheduler 
    lr_scheduler = make_lr_scheduler(cfg, optimizer)

    for epoch in range(cfg.OPTIMIZER.MAX_EPOCH):
        loader = tqdm.tqdm(train_loader)
        epoch_loss = 0.0
        iters = 0
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            classes = torch.unique(y)

            # separate data from different classes
            idx = torch.arange(len(y))
            
            support_set = []
            query_set = []

            for c in classes:
                tmp = idx[y==c]

                # if this batch contains only 1 instance for a minority class
                if len(tmp) >= 2:
                    support_idx = tmp[1:]
                    query_idx = tmp[0]
                else:
                    support_idx = query_idx = tmp

                support = x[support_idx].float()
                query = x[query_idx].view(1, -1).float()

                support = model(support)

                support_set.append(support)

                query = model(query)
                query_set.append(query)

            support_set = torch.cat(support_set, dim=0)
            query_set = torch.cat(query_set, dim=0)

            # compute prediction probability for all the queries
            logit = torch.matmul(query_set, support_set.t())

            loss = criterion(logit, classes.long())

            epoch_loss += loss.item()
            iters += 1

            loader.set_description('Epoch: {} Batch Loss={:.2f} Avg Epoch Loss={:.2f}'.format(epoch, loss.item(), epoch_loss/iters)) 

            # backward
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            lr_scheduler.step()

        # save model for every epoch 
        # TODO: figure out which model is better and save it
        if save:
            print("Saving model to {}/{}/{}".format(cfg.OUTPUT_DIR, cfg.DATASET.TRAIN, 'Epoch_{}.pth'.format(epoch)))
            torch.save(model, '{}/{}/{}'.format(cfg.OUTPUT_DIR, cfg.DATASET.TRAIN, 'Epoch_{}.pth'.format(epoch)))
        
        # model evaluation
        print("Evaluating on Validation Dataset")
        model.eval()
        f1, gmean, auc, specificity, sensitivity = test(cfg, model, valid_loader, device=cfg.DEVICE)
        # score = f1 * gmean * auc * specificity * sensitivity
        score = min(f1, gmean, auc, specificity, sensitivity)
        print("score: ", score)
        if score > best_score:
            best_epoch = epoch
            best_score = score
        model.train()
        print()
    return best_epoch

def test(cfg, model, test_loader, pos_classes=cfg.MODEL.MINORITY_CLASS, device='cpu'):
    print("Minority Classes: ", pos_classes)
    model = model.to(device)
    model.anchor = model.anchor.to(device)
    model.eval()

    with torch.no_grad():
        """post-training step"""
        cls_embedding = []
        labels = []

        # load training dataset
        x, y = load_dataset(cfg)
        # randomly sample some samples and matins the class ratio in it
        classes = np.unique(y)
        cls_ratio = {}
        for c in classes:
            cls_ratio[c] = sum(y==c) / len(y)
        train_sample_size = cfg.TEST.TRAIN_SAMPLE_SIZE
        idx = np.arange(len(y))
        remain_size = train_sample_size
        for c in classes:
            cls_idx = idx[y==c]
            np.random.shuffle(cls_idx)
            sampled_idx = cls_idx[:min(max(int(train_sample_size * cls_ratio[c]), 1), remain_size)]
            remain_size -= len(sampled_idx)

            # get embedding
            samples = x[sampled_idx]
            samples = torch.from_numpy(samples).to(device).float()
            embedding = model(samples)

            cls_embedding.append(embedding)
            labels.append(c)
        cls_embedding = torch.cat(cls_embedding, dim=0)

        """test"""
        # start evaluation on test dataset
        test_loader = tqdm.tqdm(test_loader)
        predictions = []
        ground_truths = []
        scores = []
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device).float(), y.to(device)
            emb = model(x) 

            rel_score = torch.softmax(torch.matmul(emb, cls_embedding.t()), dim=1)
            predict_cls = labels[torch.argmax(rel_score, dim=1)]

            predictions.append(predict_cls.item()) 
            ground_truths.append(y.item())
            scores.append(rel_score[0, 1])

        # compute metrics
        gmean = geometric_mean_score(ground_truths, predictions, average='binary')
        f1 = f1_score(ground_truths, predictions, average='macro')
        auc = roc_auc_score(ground_truths, scores, average='macro')
        specificity = specificity_score(ground_truths, predictions, average='binary')
        sensitivity = sensitivity_score(ground_truths, predictions, average='binary')
        print("F1: {:.5f} | G-Mean:{:.5f} | AUC: {:.5f} | Spec: {:.5f} | Sens: {:.5f}".format(f1, gmean, auc, specificity, sensitivity))
        return f1, gmean, auc, specificity, sensitivity

def main():
    parser = argparse.ArgumentParser(description='PyTorch Imbalanced Metric Learning')

    parser.add_argument(
        '--config-file',
        default='../configs/default.yaml',
        metavar='FILE',
        help='path to configuration file',
        type=str
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # create output dir
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.TRAIN)):
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.TRAIN))

    # set up logger 
    logger = setup_logger('Imbalanced', cfg.OUTPUT_DIR)

    logger.info("Collecting env info (may take some time)\n")
    logger.info(get_pretty_env_info())

    logger.info("Loading configuration file from {}".format(args.config_file))
    
    with open(args.config_file) as f:
        config_str = f.read()
        config_str = '\n' + config_str.strip()
        logger.info(config_str)
    
    logger.info('Running with configuration: \n')
    logger.info(cfg)

    # set random seed for pytorch and numpy 
    if cfg.SEED != 0:
        logger.info("Using manual seed: {}".format(cfg.SEED))
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        np.random.seed(cfg.SEED)
    else:
        logger.info("Using random seed")
        torch.backends.cudnn.benchmark = True
    
    train_loader = make_data_loader(cfg, is_train=True, is_valid=False)
    valid_loader = make_data_loader(cfg, is_train=False, is_valid=True)
    test_loader = make_data_loader(cfg, is_train=False, is_valid=False)
    model = build_model(cfg).to(cfg.DEVICE)
    logger.info(model)

    logger.info("Start Training ...")
    best_epoch = train(cfg, model, train_loader, valid_loader, save=True)
    logger.info("Done.")
    
    # load best model on validation dataset
    # please specify the best model, by default, it is the one with best G-Mean
    best_name = 'Epoch_{}'.format(best_epoch)
    best_model = torch.load('{}/{}/{}'.format(cfg.OUTPUT_DIR, cfg.DATASET.TRAIN, '{}.pth'.format(best_name)))
    logger.info("Best Model Name: {} - Start Evaluation ...".format(best_name))
    test(cfg, best_model, test_loader)
    logger.info("Done.")

if __name__ == '__main__':
    main()
