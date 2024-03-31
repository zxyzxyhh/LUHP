import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import torchvision
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
sys.path.append('../../..')
from examples.domain_adaptation.image_classification import na_utils
import numpy as np
from examples.domain_adaptation.image_classification import LUHP
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

sys.path.append('.')
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def EculideanDistances(a, b):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    sq_a = a**2
    sq_b = b**2
    sq_a_sum = torch.sum(sq_a, dim=1).unsqueeze(1)
    sq_b_sum = torch.sum(sq_b, dim=1).unsqueeze(0)
    bt = b.t()
    return torch.sqrt(sq_a_sum + sq_b_sum - 2*a.mm(bt))
def data_load(args):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if args.source == ['A']:
        s_dset_path = "D:\\data\\office31\\amazon\\amazon.txt"
    elif args.source == ['D']:
        s_dset_path = "D:\\data\\office31\\dslr\\dslr.txt"
    else:
        s_dset_path = "D:\\data\\office31\\webcam\\webcam.txt"

    if args.target == ['A']:
        t_dset_path = "D:\\data\\office31\\amazon\\amazon.txt"
    elif args.target == ['D']:
        t_dset_path = "D:\\data\\office31\\dslr\\dslr.txt"
    else:
        t_dset_path = "D:\\data\\office31\\webcam\\webcam.txt"
    print(s_dset_path)
    print(t_dset_path)
    source_set = na_utils.ObjectImage_mul('', s_dset_path, train_transform)
    target_set = na_utils.ObjectImage_mul('', t_dset_path, train_transform)
    test_set = na_utils.ObjectImage('', t_dset_path, test_transform)

    dset_loaders = {}
    dset_loaders["source"] = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, drop_last=True)
    dset_loaders["target"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, drop_last=True)
    dset_loaders["test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.workers, drop_last=False)
    return dset_loaders
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss
def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True
    #creat data
    dset_loaders = data_load(args)
    train_source_iter = iter(dset_loaders["source"])
    train_target_iter = iter(dset_loaders["target"])
    val_loader = test_loader = dset_loaders["test"]
    num_classes = 31
    mem_fea_target = torch.rand(len(dset_loaders["target"].dataset), 256).to(device)
    mem_fea_target = mem_fea_target / torch.norm(mem_fea_target, p=2, dim=1, keepdim=True)
    mem_cls_target = torch.rand(len(dset_loaders["target"].dataset), 31).to(device)
    mem_weight_target = torch.ones(len(dset_loaders["target"].dataset),).to(device)
    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))#+ domain_discri.get_parameters()

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_iter, feature_extractor, device)
        target_feature = collect_feature(train_source_iter, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    luhp = LUHP.MemoryTripletK_reuse(31, len(dset_loaders["target"].dataset))
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        mem_fea_target, mem_cls_target, mem_weight_target = train(dset_loaders,  train_source_iter, train_target_iter, classifier, domain_adv, optimizer,
              lr_scheduler, epoch, args, mem_fea_target, mem_cls_target, mem_weight_target, luhp)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(dset_loaders, train_source_iter, train_target_iter,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, mem_fea_target, mem_cls_target, mem_weight_target, luhp):

    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':9.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':9.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    class_weight_src = torch.ones(31, ).to(device)
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        try:
            x_s, labels_s, idx_s = next(train_source_iter)
        except:
            train_source_iter = iter(dset_loaders["source"])
            x_s, labels_s, idx_s = next(train_source_iter)
        try:
            x_t, labels_t, idx_t = next(train_target_iter)
        except:
            train_target_iter = iter(dset_loaders["target"])
            x_t, labels_t, idx_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        # data_time.update(time.time() - end)
        eff = (i + args.iters_per_epoch * epoch) / (args.iters_per_epoch * args.epochs)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        f_t_n = f_t / torch.norm(f_t, p=2, dim=1, keepdim=True)
        softmax_out = nn.Softmax(dim=1)(y_t)
        outputs_target = softmax_out ** 2 / ((softmax_out ** 2).sum(dim=0))

        src = CrossEntropyLabelSmooth(num_classes=31, epsilon=args.smooth)(y_s, labels_s)
        weight_src = class_weight_src[labels_s].unsqueeze(0)
        cls_loss = torch.sum(weight_src * src) / (torch.sum(weight_src).item())
        transfer_loss = domain_adv(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        dis_1 = EculideanDistances(f_t.detach(), mem_fea_target)
        for di in range(dis_1.size(0)):
            dis_1[di, idx_t[di]] = torch.max(dis_1)
        v1, p1 = torch.sort(dis_1, dim=1)
        w = torch.zeros(f_t.size(0), mem_fea_target.size(0)).cuda()
        w_1 = torch.zeros(f_t.size(0), mem_fea_target.size(0)).cuda()
        for wi in range(w.size(0)):
            for wj in range(args.K):
                w_1[wi][p1[wi, wj]] = 1.0 / args.K

        weight, pred = torch.max(w_1.mm(mem_cls_target), 1)
        luhp_loss, re_loss, W = luhp(f_t, pred, idx_t, mem_fea_target, mem_cls_target, weight, x_t, model, softmax_out)
        self_loss = nn.CrossEntropyLoss(reduction='none')(y_t, pred)
        self_loss = torch.sum(weight * self_loss) / (torch.sum(weight).item())
        loss = cls_loss + args.self_weight * eff * self_loss + transfer_loss * args.trade_off
        if luhp.iteration > luhp.start_check_noise_iteration:
            loss = loss + luhp_loss * args.luhp_weight * eff
        loss = loss + re_loss * args.self_weight * eff
        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]
        losses.update(luhp_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        end = time.time()
        with torch.no_grad():
            mem_fea_target[idx_t] = f_t_n.clone()
            mem_cls_target[idx_t] = outputs_target.clone()
            mem_weight_target[idx_t] = W.clone()
        if i % args.print_freq == 0:
            progress.display(i)
    return mem_fea_target, mem_cls_target, mem_weight_target
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LUHP for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office-home', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office-home)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=24, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-9)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--lambdas', type=float, default=0.1)
    parser.add_argument('--luhp_weight', type=float, default=1.0)
    parser.add_argument('--self_weight', type=float, default=0.2)
    args = parser.parse_args()
    main(args)

