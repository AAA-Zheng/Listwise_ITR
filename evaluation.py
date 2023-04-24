from __future__ import print_function
import os
import pickle

import numpy
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary, deserialize_vocab  # NOQA
import torch
from model import VSE
from collections import OrderedDict

import json
from eccv_caption import Metrics


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            images, captions, lengths, image_ids, caption_ids, v_text_emb, t_text_emb = batch_data
            # make sure val logger is used
            model.logger = val_logger

            # compute the embeddings
            img_emb, cap_emb = model.forward_emb(images, captions, lengths)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = np.zeros(
                    (len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros(
                    (len(data_loader.dataset), cap_emb.size(1)))

            # preserve the embeddings by copying from GPU
            # and converting to NumPy
            img_embs[caption_ids] = img_emb.data.cpu().numpy().copy()
            cap_embs[caption_ids] = cap_emb.data.cpu().numpy().copy()

            # # measure accuracy and record loss
            # model.forward_loss(img_emb, cap_emb, ids, v_bert_emb, t_bert_emb)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        .format(
                            i, len(data_loader), batch_time=batch_time,
                            e_log=str(model.logger)))
            del images, captions

    return img_embs, cap_embs


def evalrank(model_path, data_path=None, split='dev', fold5=False, save_path=None):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        start = time.time()

        sims = compute_sim(img_embs, cap_embs)
        npts = img_embs.shape[0]

        end = time.time()
        print("calculate similarity time: {}".format(end - start))

        if save_path is not None:
            np.save(save_path, {'npts': npts, 'sims': sims})
            print('Save the similarity into {}'.format(save_path))

        # no cross-validation, full evaluation
        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)

        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
        print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            start = time.time()
            sims = compute_sim(img_embs_shard, cap_embs_shard)
            end = time.time()
            print("calculate similarity time: {}".format(end - start))

            npts = img_embs_shard.shape[0]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(npts, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f" % rsum)
            results += [list(r) + list(ri) + [rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % mean_metrics[10])
        print("Image to text: %.1f, %.1f, %.1f, %.1f %.1f" %
              mean_metrics[:5])
        print("Text to image: %.1f, %.1f, %.1f, %.1f %.1f" %
              mean_metrics[5:10])


def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def evalrank_eccv(model_path, data_path=None, split='test'):

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    cxc_annot_base = os.path.join('cxc_annots')
    img_id_path = os.path.join(cxc_annot_base, 'testall_ids.txt')
    cap_id_path = os.path.join(cxc_annot_base, 'testall_capids.txt')

    with open(img_id_path) as f:
        img_ids = f.readlines()
    with open(cap_id_path) as f:
        cap_ids = f.readlines()

    img_ids = [int(img_id.strip()) for i, img_id in enumerate(img_ids) if i % 5 == 0]
    cap_ids = [int(cap_id.strip()) for cap_id in cap_ids]

    img_ids2index = {}
    for index, img_id in enumerate(img_ids):
        img_ids2index[img_id] = index

    cap_ids2index = {}
    for index, cap_id in enumerate(cap_ids):
        cap_ids2index[cap_id] = index

    metric = Metrics()

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    sims = compute_sim(img_embs, cap_embs)
    sims = torch.Tensor(sims).cuda()

    i2t_rank = {}
    t2i_rank = {}

    all_cids = np.array(cap_ids)
    all_iids = np.array(img_ids)

    K = 100
    for idx, iid in enumerate(all_iids):
        values, indices = sims[idx, :].topk(K)
        indices = indices.detach().cpu().numpy()
        i2t_rank[iid] = [int(cid) for cid in all_cids[indices]]

    for idx, cid in enumerate(all_cids):
        values, indices = sims[:, idx].topk(K)
        indices = indices.detach().cpu().numpy()
        t2i_rank[cid] = [int(iid) for iid in all_iids[indices]]

    scores = metric.compute_all_metrics(
        i2t_rank, t2i_rank,
        target_metrics=['eccv_r1', 'eccv_map_at_r', 'eccv_rprecision',
                        'coco_1k_recalls', 'coco_5k_recalls',
                        'cxc_recalls'
                        ],
        Ks=[1, 5, 10],
        verbose=False
    )
    # for key in scores:
    #     print(key, scores[key])

    print('-------------------------------------')
    rsum = 100 * (scores['coco_1k_r1']['i2t'] + scores['coco_1k_r5']['i2t'] + scores['coco_1k_r10']['i2t']
                  + scores['coco_1k_r1']['t2i'] + scores['coco_1k_r5']['t2i'] + scores['coco_1k_r10']['t2i'])
    print('COCO 1K RSUM: %.1f' % rsum)
    print('Image to text: %.1f, %.1f, %.1f' %
          (scores['coco_1k_r1']['i2t'] * 100, scores['coco_1k_r5']['i2t'] * 100, scores['coco_1k_r10']['i2t'] * 100))
    print('Text to image: %.1f, %.1f, %.1f' %
          (scores['coco_1k_r1']['t2i'] * 100, scores['coco_1k_r5']['t2i'] * 100, scores['coco_1k_r10']['t2i'] * 100))

    print('-------------------------------------')
    rsum = 100 * (scores['coco_5k_r1']['i2t'] + scores['coco_5k_r5']['i2t'] + scores['coco_5k_r10']['i2t']
                  + scores['coco_5k_r1']['t2i'] + scores['coco_5k_r5']['t2i'] + scores['coco_5k_r10']['t2i'])
    print('COCO 5K RSUM: %.1f' % rsum)
    print('Image to text: %.1f, %.1f, %.1f' %
          (scores['coco_5k_r1']['i2t'] * 100, scores['coco_5k_r5']['i2t'] * 100, scores['coco_5k_r10']['i2t'] * 100))
    print('Text to image: %.1f, %.1f, %.1f' %
          (scores['coco_5k_r1']['t2i'] * 100, scores['coco_5k_r5']['t2i'] * 100, scores['coco_5k_r10']['t2i'] * 100))

    print('-------------------------------------')
    eccv_sum = 100 * (scores['eccv_map_at_r']['i2t'] + scores['eccv_rprecision']['i2t'] + scores['eccv_r1']['i2t']
                      + scores['eccv_map_at_r']['t2i'] + scores['eccv_rprecision']['t2i'] + scores['eccv_r1']['t2i'])
    print('ECCV SUM: %.1f' % eccv_sum)
    print('Image to text: %.2f, %.2f, %.2f' %
          (
              scores['eccv_map_at_r']['i2t'] * 100, scores['eccv_rprecision']['i2t'] * 100,
              scores['eccv_r1']['i2t'] * 100))
    print('Text to image: %.2f, %.2f, %.2f' %
          (
              scores['eccv_map_at_r']['t2i'] * 100, scores['eccv_rprecision']['t2i'] * 100,
              scores['eccv_r1']['t2i'] * 100))



def i2t(npts, sims, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def eval_ensemble(results_paths, fold5=False):
    all_sims = []
    all_npts = []
    for sim_path in results_paths:
        results = np.load(sim_path, allow_pickle=True).tolist()
        npts = results['npts']
        sims = results['sims']
        all_npts.append(npts)
        all_sims.append(sims)
    all_npts = np.array(all_npts)
    all_sims = np.array(all_sims)
    assert np.all(all_npts == all_npts[0])
    npts = int(all_npts[0])
    sims = all_sims.mean(axis=0)

    if not fold5:
        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        # print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        # print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        npts = npts // 5
        results = []
        all_sims = sims.copy()
        for i in range(5):
            sims = all_sims[i * npts: (i + 1) * npts, i * npts * 5: (i + 1) * npts * 5]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(npts, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]
        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[12]))
        # print("Average i2t Recall: %.1f" % mean_metrics[10])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[:5])
        # print("Average t2i Recall: %.1f" % mean_metrics[11])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[5:10])
