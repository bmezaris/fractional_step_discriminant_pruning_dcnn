"""
Implementation of FSDP described in the paper:
N. Gkalelis, V. Mezaris, "Fractional Step Discriminant Pruning:
A Filter Pruning Framework for Deep Convolutional Neural Networks",
Proc. 7th IEEE Int. Workshop on Mobile Multimedia Computing (MMC2020)
at the IEEE Int. Conf. on Multimedia and Expo (ICME), London, UK, July 2020.
History
-------
DATE       | DESCRIPTION                   | NAME              | ORGANIZATION |
16/01/2020 | first creation of FSDP method | Nikolaos Gkalelis | CERTH-ITI    |
"""

from __future__ import division

import os, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
import numpy as np
import pickle
from scipy.spatial import distance
from torch.nn.functional import one_hot, relu

from utilities.expPrmDecay import cmpAsymptoticSchedule

from torch.utils.data.sampler import WeightedRandomSampler

import speech_transforms
from datasets.gsc import gsc_utils

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['gsc'], help='Dataset to use.')
parser.add_argument('--arch', metavar='ARCH', default='resnet20', choices=['resnet20', 'resnet56', 'resnet110' ])
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[1, 60, 120, 160], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[10, 0.2, 0.2, 0.2],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--layer_begin', type=int, default=0, help='index of first conv layer of model')
parser.add_argument('--layer_end', type=int, default=54, help='index of last conv layer of model')
parser.add_argument('--layer_inter', type=int, default=3, help='interval between conv layers in the model')
parser.add_argument('--epoch_prune', type=int, default=1, help='every how many epochs to prune the model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')
parser.add_argument('--pretrain_path', default='', type=str, help='..path of pre-trained model')

parser.add_argument('--batch_prune_size', type=int, default=256, help='Batch size.')
parser.add_argument('--use_zero_scaling', dest='use_zero_scaling', action='store_true', help='use zero scaling factors or asymptotic')
parser.add_argument('--max_iter_cs', type=int, default=10000, help='maximum number of batch iterations for cs criterion')
parser.add_argument('--epoch_apply_cs', type=int, default=range(0, 200, 2), help='epochs to apply the cs criterion')
parser.add_argument('--tau', type=float, default=10., help='parameter tau for computing the asymptotic pruning schedule')
parser.add_argument('--prune_rate_cs', type=float, default=0.1, help='final pruning rate for cs criterion')
parser.add_argument('--prune_rate_gm', type=float, default=0.4, help='the reducing ratio of pruning based on Distance')

parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
parser.add_argument('--multi_crop', action='store_true', help='apply crop and average the results')

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True


def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("CS Pruning Rate: {}".format(args.prune_rate_cs), log)
    print_log("GM Pruning Rate: {}".format(args.prune_rate_gm), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("use pretrain: {}".format(args.use_pretrain), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    n_mels = 32
    if args.input == 'mel40':
        n_mels = 40

    data_aug_transform = transforms.Compose([
        speech_transforms.ChangeAmplitude(),
        speech_transforms.ChangeSpeedAndPitchAudio(),
        speech_transforms.FixAudioLength(),
        speech_transforms.ToSTFT(),
        speech_transforms.StretchAudioOnSTFT(),
        speech_transforms.TimeshiftAudioOnSTFT(),
        speech_transforms.FixSTFTDimension()])

    parser.add_argument("--background_noise", type=str, default='datasets/gsc/train/_background_noise_',
                        help='path of background noise')

    backgroundNoisePname = os.path.join(args.data_path, 'train\_background_noise_')
    bg_dataset = gsc_utils.BackgroundNoiseDataset(backgroundNoisePname, data_aug_transform)
    add_bg_noise = speech_transforms.AddBackgroundNoiseOnSTFT(bg_dataset)

    train_feature_transform = transforms.Compose([
        speech_transforms.ToMelSpectrogramFromSTFT(n_mels=n_mels),
        speech_transforms.DeleteSTFT(),
        speech_transforms.ToTensor('mel_spectrogram', 'input')])

    train_dataset = gsc_utils.SpeechCommandsDataset(
        os.path.join(args.data_path, 'train'),
        transforms.Compose([speech_transforms.LoadAudio(),
                            data_aug_transform,
                            add_bg_noise,
                            train_feature_transform]))

    valid_feature_transform = transforms.Compose([
        speech_transforms.ToMelSpectrogram(n_mels=n_mels),
        speech_transforms.ToTensor('mel_spectrogram', 'input')])
    valid_dataset = gsc_utils.SpeechCommandsDataset(
        os.path.join(args.data_path, 'valid'),
        transforms.Compose([
            speech_transforms.LoadAudio(),
            speech_transforms.FixAudioLength(),
            valid_feature_transform]))
    test_dataset = gsc_utils.SpeechCommandsDataset(
        os.path.join(args.data_path, 'test'),
        transforms.Compose([
            speech_transforms.LoadAudio(),
            speech_transforms.FixAudioLength(),
            valid_feature_transform]),
        silence_percentage=0)


    weights = train_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                               num_workers=args.workers, pin_memory=True)
    train_prune_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_prune_size,
                                                     num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    num_classes = len(gsc_utils.CLASSES)



    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes=num_classes, in_channels=1, fctMultLinLyr=64)
    print_log("=> network :\n {}".format(net), log)

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    if args.use_pretrain:
        pretrain = torch.load(args.pretrain_path)
        if args.use_state_dict:
            net.load_state_dict(pretrain['state_dict'])
        else:
            net = pretrain['state_dict']

    recorder = RecorderMeter(args.epochs)

    mdlIdx2ConvIdx = [] # module index to conv filter index
    for index1, layr in enumerate(net.modules()):
        if isinstance(layr, torch.nn.Conv2d):
            mdlIdx2ConvIdx.append(index1)

    prmIdx2ConvIdx = [] # parameter index to conv filter index
    for index2, item in enumerate(net.parameters()):
        if len(item.size()) == 4:
            prmIdx2ConvIdx.append(index2)

    # set index of last layer depending on the known architecture
    if args.arch == 'resnet20':
        args.layer_end = 54
    elif args.arch == 'resnet56':
        args.layer_end = 162
    elif args.arch == 'resnet110':
        args.layer_end = 324
    else:
        pass # unkonwn architecture, use input value

    # asymptotic schedule
    total_pruning_rate = args.prune_rate_gm + args.prune_rate_cs
    compress_rates_total, scalling_factors, compress_rates_cs, compress_rates_fpgm, e2 =\
        cmpAsymptoticSchedule(theta3=total_pruning_rate, e3=args.epochs-1, tau=args.tau, theta_cs_final = args.prune_rate_cs) # tau=8.
    keep_rate_cs = 1. - compress_rates_cs

    if args.use_zero_scaling:
        scalling_factors = np.zeros(scalling_factors.shape)

    m = Mask(net, train_prune_loader, mdlIdx2ConvIdx, prmIdx2ConvIdx, scalling_factors, keep_rate_cs, compress_rates_fpgm, args.max_iter_cs)
    m.set_curr_epoch(0)
    m.set_epoch_cs(args.epoch_apply_cs)
    m.init_selected_filts()
    m.init_length()

    val_acc_1, val_los_1 = validate(test_loader, net, criterion, log)

    m.model = net
    m.init_mask(keep_rate_cs[0], compress_rates_fpgm[0], scalling_factors[0])
    #    m.if_zero()
    m.do_mask()
    m.do_similar_mask()
    net = m.model
    #    m.if_zero()
    if args.use_cuda:
        net = net.cuda()
    val_acc_2, val_los_2 = validate(test_loader, net, criterion, log)

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(args.start_epoch, args.epochs):

        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log, m)
        
        # evaluate on validation set
        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:
            m.model = net
            m.set_curr_epoch(epoch)
            m.if_zero()
            m.init_mask(keep_rate_cs[epoch], compress_rates_fpgm[epoch], scalling_factors[epoch])
            m.do_mask()
            m.do_similar_mask()
            m.if_zero()
            net = m.model
            if args.use_cuda:
                net = net.cuda()
            if epoch == args.epochs - 1:
                m.if_zero()

        val_acc_2, val_los_2 = validate(test_loader, net, criterion, log)

        is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

    log.close()


# train function
def train(train_loader, model, criterion, optimizer, epoch, log, m):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, aBatch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = aBatch['target']
        input = aBatch['input']
        input = torch.unsqueeze(input, 1)

        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Mask grad for iteration
        m.do_grad_mask()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg),
        log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, aBatch in enumerate(val_loader):

        target = aBatch['target']
        input = aBatch['input']
        input = torch.unsqueeze(input, 1)

        N = input.size(0) # number of observations in the batch
        if args.multi_crop:
            input = gsc_utils.multi_crop(input)

        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        output = model(input)

        if args.multi_crop:
            output = output.view(-1, N, input.size(1))
            output = torch.mean(output, dim=0)
            output = torch.nn.functional.softmax(output, dim=1)

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

    print_log('  **Test** Prec@1 {top1:.3f} Prec@5 {top5:.3f} Error@1 {error1:.3f}'.format(
        top1=top1.avg, top5=top5.avg, error1=100 - top1.avg), log)

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Mask:
    def __init__(self, model, train_prune_loader, mdlIdx2ConvIdx, prmIdx2ConvIdx, zetas, keep_rate_cs, compress_rate_fpgm, max_iter_cs):
        self.model_size = {}
        self.model_length = {}
        self.cs_rate = {}
        self.gm_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}

        self.train_prune_loader = train_prune_loader
        self.mdlIdx2ConvIdx = np.array(mdlIdx2ConvIdx)
        self.prmIdx2ConvIdx = np.array(prmIdx2ConvIdx)
        self.max_iter_cs = max_iter_cs
        self.NN = self._cmpInvCardinalityMat()
        self.alpha_curr = 0.
        self.curr_epoch = 0 # current epoch
        self.fselected_prune = {}  # selected filters to attenuate and discard at the end
        self.filter_index_sorted = {}
        self.zetas = zetas
        self.keep_rate_cs = keep_rate_cs
        self.keep_rate_gm = compress_rate_fpgm
        self.epoch_cs = []

    def computeDiscrScores(self):

        print('CS criterion>> Entering')

        time1 = time.time()

        num_classes = list(self.model.modules())[-1].out_features

        # initialize data containers and hooks to obtain layer feature map output
        XXX = []  # feature map
        hh = []  # hook handler

        conv_size = []
        layer2index = {}
        index2layer = {}

        j=0
        for index, LayerPrms in enumerate(self.model.parameters()):
            if index in self.mask_index: # if conv filter
                conv_size.append(LayerPrms.size()[0])
                layer2index[j] = index
                index2layer[index] = j
                j += 1

                def _conv_layer_hook_function(module, input_, output):
                    nonlocal XXX
                    XX, _ = torch.max(relu(output.clone().detach()), dim=3) # ensure ReLu is applied
                    XX, _ = torch.max(XX, dim=2)
                    XXX.append(XX)

                mdlFiltIdx = self.mdlIdx2ConvIdx[self.prmIdx2ConvIdx == index]  # get filter index in module level
                for indexLayr, layr in enumerate(self.model.modules()):
                    if indexLayr == mdlFiltIdx + 1:  # get the batch normalized data (one layer next)
                        #hh[indexLayr] = layr.register_forward_hook(_conv_layer_hook_function)
                        hh.append( layr.register_forward_hook(_conv_layer_hook_function) )

        L = len(conv_size) # number of layers
        fmx = max(conv_size)  # maximum number of filters in a layer


        RXX = torch.zeros((L, fmx, num_classes, 1), dtype=torch.double).cuda()

        niter = 0
        for aBatch in self.train_prune_loader:
            X = aBatch['input']
            Y = aBatch['target']
            X = torch.unsqueeze(X, 1)

            X = X.cuda()
            Y = Y.cuda()
            Li = list(Y.size())[0]  # number of observations may be less than batch size in the last batch!
            _ = self.model(X)  # fill hooks
            R = one_hot(Y, num_classes=num_classes).float()  # compute indicator matrix

            for l in range(L): # for each layer
                F = conv_size[l]
                for f in range(F):  # for each filter in this layer
                    RXX[l,f,:,:] += torch.mm(torch.t(R), XXX[l][:, f].unsqueeze_(1))
                    #print('CS criterion, processing Layer - filter idx : {} - {}'.format(l, f))

            while XXX: # clear for next iteration
                aX = XXX.pop(0)
                del aX

            niter += 1
            #print('CS criterion, iteration : {}'.format(niter))
            if niter == self.max_iter_cs:
                break

        XXX.clear() # clear for next itration
        XXX = None
        del XXX

        while hh:
            ahh = hh.pop(0)
            ahh.remove()
            del ahh

        #for hhkey in hh:
        #    hh[hhkey].remove()
        #    hh[hhkey] = None

        # compute discriminant score for each filter
        dnp = np.zeros((L, fmx), dtype=np.float64)
        NN = self.NN.double()
        for l in range(L):  # for each layer
            F = conv_size[l]
            for f in range(F):  # for each filter in this layer
                RXf = torch.t(RXX[l,f,:,:]).double()
                M = torch.mm(RXf, NN)  # compute mean matrix
                P = num_classes * torch.eye(num_classes, dtype=torch.float64)
                P -= torch.ones([num_classes, num_classes], dtype=torch.float64)
                P = P.cuda()
                Sb = torch.mm(torch.mm(M, P), torch.t(M))
                dnp[l, f] = torch.trace(Sb).detach().cpu().numpy()

        time2 = time.time()
        #print('CS criterion>> Exiting; Time needed (secs) {}'.format( (time2 - time1) ))

        return dnp, index2layer, conv_size

    def _cmpInvCardinalityMat(self):

        num_classes = list(self.model.modules())[-1].out_features

        # compute class cardinality vector and inverse cardinality matrix matrix
        nn = torch.zeros([num_classes, ], dtype=torch.int64)  # initialize class cardinality vector
        niter = 0
        for bidx, aBatch in enumerate(self.train_prune_loader):
            labels = aBatch['target']
            ll, nn_i = torch.unique(labels, sorted=True, return_counts=True)
            if len(ll) != num_classes:
                for idx, lbl in enumerate(
                        ll):  # assuming integer labels in ascending order starting from 0 to num_classes-1
                    nn[int(lbl)] = nn_i[idx]
            else:
                nn += nn_i
            niter += 1
            if niter == self.max_iter_cs:
                break


        nn = nn.float()
        # N = torch.sum(nn)  # total number of training observations
        NN = torch.diag(torch.tensor(1.) / nn).cuda()

        return NN

    # optimize for fast ccalculation
    def get_filter_similar(self, index, weight_torch, compress_rate_gm, keep_rate_cs, scaling_factor, filter_index_sorted_da, length):

        print('Pruning rates: CS {} GM {}'.format(1- keep_rate_cs, compress_rate_gm))

        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - keep_rate_cs))
            similar_pruned_num = int(weight_torch.size()[0] * compress_rate_gm)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            filter_large_index = filter_index_sorted_da[filter_pruned_num:]

            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_cs = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            similar_matrix = distance.cdist(weight_vec_after_cs, weight_vec_after_cs, 'euclidean')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            print('GM criterion, selected filters | Layer idx - prune filter idx - alpha: {} - {} - {}'.format(
                index, similar_index_for_filter, scaling_factor))

            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = scaling_factor

        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, compress_rate_cs_per_layer, compress_rate_gm_per_layer):
        for index, item in enumerate(self.model.parameters()):
            self.cs_rate[index] = 1
            self.gm_rate[index] = 1
        for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
            self.cs_rate[key] = compress_rate_cs_per_layer
            self.gm_rate[key] = compress_rate_gm_per_layer
        # different setting for  different architecture
        if args.arch == 'resnet20':
            last_index = 57
        elif args.arch == 'resnet56':
            last_index = 165
        elif args.arch == 'resnet110':
            last_index = 327
        # to jump the last fc layer
        self.mask_index = [x for x in range(0, last_index, 3)]

    #        self.mask_index =  [x for x in range (0,330,3)]

    def init_mask(self, keep_rate_cs, compress_rate_fpgm, scaling_factor):

        print('Pruning rates: CS {} GM {}'.format(1 - keep_rate_cs, compress_rate_fpgm))

        self.init_rate(keep_rate_cs, compress_rate_fpgm)

        dnp = []
        index2layer = []
        conv_size = []
        if self.curr_epoch == 0 or (self.curr_epoch in self.epoch_cs):
            dnp, index2layer, conv_size = self.computeDiscrScores()

        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:  # if conv filter

                # mask for cs criterion
                prmFiltIdx = index
                weight_torch = item.data
                length = self.model_length[index]

                codebook = np.ones(length)
                filter_pruned_num = int(weight_torch.size()[0] * (1 - keep_rate_cs))
                filter_index_sorted = []
                if len(weight_torch.size()) == 4:  # and filter_pruned_num > 0:
                    filter_index_selected = []
                    if self.curr_epoch == 0 or (self.curr_epoch in self.epoch_cs):

                        ll = index2layer[prmFiltIdx]
                        filter_index_sorted = dnp[ll, :conv_size[ll]].argsort()

                        filter_index_selected = list(filter_index_sorted[:filter_pruned_num])
                        self.fselected_prune[prmFiltIdx] = filter_index_selected
                        self.filter_index_sorted[prmFiltIdx] = filter_index_sorted
                    else:
                        filter_index_selected = self.fselected_prune[prmFiltIdx]
                        filter_index_sorted = self.filter_index_sorted[prmFiltIdx]

                    print('CS criterion, selected filters | Layer idx - prune filter idx - alpha: {} - {} - {}'.format(
                        prmFiltIdx, filter_index_selected, scaling_factor))

                    if filter_pruned_num > 0:
                        kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
                        for x in range(0, len(filter_index_selected)):
                            codebook[filter_index_selected[x] * kernel_length: (filter_index_selected[
                                                                                    x] + 1) * kernel_length] = scaling_factor  # * codebook[filter_index_selected[x] * kernel_length: (filter_index_selected[x] + 1) * kernel_length]

                self.mat[index] = codebook
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()

                # mask for distance criterion
                self.similar_matrix[index] = self.get_filter_similar(index, item.data, compress_rate_fpgm,
                                                                     keep_rate_cs, scaling_factor,
                                                                     filter_index_sorted,
                                                                     self.model_length[index])
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if args.use_cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        print("mask similar Done")

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                b = a * self.mat[index]
                b = b * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])
        # print("grad zero Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print( "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))


    def set_train_prune_loader(self, train_prune_loader):
        self.train_prune_loader = train_prune_loader

    def set_curr_epoch(self, aEpoch):
        self.curr_epoch = aEpoch  # current epoch

    def init_selected_filts(self):
        for index, item in enumerate(self.model.parameters()):
            self.fselected_prune[index] = []  # selected filters to attenuate and discard at the end
            self.filter_index_sorted[index] = []

    def set_epoch_cs(self, aEpoch):
        self.epoch_cs = aEpoch  # current epoch

if __name__ == '__main__':
    main()
