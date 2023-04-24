
"""VSE model"""

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from torch.nn.functional import avg_pool1d, max_pool1d


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImage(nn.Module):

    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.embed_size = opt.embed_size
        self.fc = nn.Linear(opt.img_dim, opt.embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)
        features = features.permute(0, 2, 1)
        features = max_pool1d(features, features.size(2)).squeeze(2)
        features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.embed_size = opt.embed_size
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, batch_first=True)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out, dim=-1)

        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def l2norm_3d(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=2, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def sigmoid(tensor, tau=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / tau
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class TripletLoss(nn.Module):

    def __init__(self, opt):
        super(TripletLoss, self).__init__()
        self.margin = opt.margin
        self.batch_size = opt.batch_size
        self.pos_mask = torch.eye(self.batch_size).cuda()
        self.neg_mask = 1 - self.pos_mask

    def forward(self, v, t):

        batch_size = v.size(0)

        scores = cosine_sim(v, t)
        pos_scores = scores.diag().view(batch_size, 1)
        pos_scores_t = pos_scores.expand_as(scores)
        pos_scores_v = pos_scores.t().expand_as(scores)

        if batch_size != self.batch_size:
            pos_mask = torch.eye(scores.size(0))
            pos_mask = pos_mask.cuda()
            neg_mask = 1 - pos_mask
        else:
            neg_mask = self.neg_mask

        loss_t = (scores - pos_scores_t + self.margin).clamp(min=0)
        loss_v = (scores - pos_scores_v + self.margin).clamp(min=0)
        loss_t = loss_t * neg_mask
        loss_v = loss_v * neg_mask
        loss_t = loss_t.max(dim=1)[0]
        loss_v = loss_v.max(dim=0)[0]
        loss_t = loss_t.mean()
        loss_v = loss_v.mean()
        loss = (loss_t + loss_v) / 2

        return loss


class SNDCGLoss(nn.Module):

    def __init__(self, opt):

        super(SNDCGLoss, self).__init__()
        self.tau = opt.tau
        self.batch_size = opt.batch_size
        self.pos_mask = torch.eye(self.batch_size).cuda()
        self.neg_mask = 1 - self.pos_mask

    def forward(self, v, t, v_text_emb, t_text_emb):

        batch_size = v.size(0)

        scores = cosine_sim(v, t)

        if batch_size != self.batch_size:
            pos_mask = torch.eye(scores.size(0))
            pos_mask = pos_mask.cuda()
            neg_mask = 1 - pos_mask
        else:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        # calculate relevance score
        v_text_emb = v_text_emb.cuda()
        t_text_emb = t_text_emb.cuda()

        v_text_emb = v_text_emb.transpose(0, 1)
        t_text_emb = t_text_emb.view(1, t_text_emb.size(0), t_text_emb.size(1))
        t_text_emb = t_text_emb.expand(5, t_text_emb.size(1), t_text_emb.size(2))

        v_text_emb = l2norm_3d(v_text_emb)
        t_text_emb = l2norm_3d(t_text_emb)
        relevance = torch.bmm(v_text_emb, t_text_emb.transpose(1, 2))
        relevance = relevance.max(0)[0]

        # norm
        relevance = (1 + relevance) / 2  # [0, 1]
        relevance = relevance * neg_mask + pos_mask

        # IDCG
        relevance_repeat = relevance.unsqueeze(dim=2).repeat(1, 1, relevance.size(0))
        relevance_repeat_trans = relevance_repeat.permute(0, 2, 1)
        relevance_diff = relevance_repeat_trans - relevance_repeat
        relevance_indicator = torch.where(relevance_diff > 0,
                                          torch.full_like(relevance_diff, 1),
                                          torch.full_like(relevance_diff, 0))
        relevance_rk = torch.sum(relevance_indicator, dim=-1) + 1
        idcg = (2 ** relevance - 1) / torch.log2(1 + relevance_rk)
        idcg = torch.sum(idcg, dim=-1)

        # scores diff
        scores_repeat_t = scores.unsqueeze(dim=2).repeat(1, 1, scores.size(0))
        scores_repeat_trans_t = scores_repeat_t.permute(0, 2, 1)
        scores_diff_t = scores_repeat_trans_t - scores_repeat_t

        scores_repeat_v = scores.t().unsqueeze(dim=2).repeat(1, 1, scores.size(0))
        scores_repeat_trans_v = scores_repeat_v.permute(0, 2, 1)
        scores_diff_v = scores_repeat_trans_v - scores_repeat_v

        # image-to-text
        scores_sg_t = sigmoid(scores_diff_t, tau=self.tau)

        scores_sg_t = scores_sg_t * neg_mask
        scores_rk_t = torch.sum(scores_sg_t, dim=-1) + 1

        scores_indicator_t = torch.where(scores_diff_t > 0,
                                         torch.full_like(scores_diff_t, 1),
                                         torch.full_like(scores_diff_t, 0))
        real_scores_rk_t = torch.sum(scores_indicator_t, dim=-1) + 1

        dcg_t = (2 ** relevance - 1) / torch.log2(1 + scores_rk_t)
        dcg_t = torch.sum(dcg_t, dim=-1)

        real_dcg_t = (2 ** relevance - 1) / torch.log2(1 + real_scores_rk_t)
        real_dcg_t = torch.sum(real_dcg_t, dim=-1)

        # text-to-image
        scores_sg_v = sigmoid(scores_diff_v, tau=self.tau)

        scores_sg_v = scores_sg_v * neg_mask
        scores_rk_v = torch.sum(scores_sg_v, dim=-1) + 1

        scores_indicator_v = torch.where(scores_diff_v > 0,
                                         torch.full_like(scores_diff_v, 1),
                                         torch.full_like(scores_diff_v, 0))
        real_scores_rk_v = torch.sum(scores_indicator_v, dim=-1) + 1

        dcg_v = (2 ** relevance - 1) / torch.log2(1 + scores_rk_v)
        dcg_v = torch.sum(dcg_v, dim=-1)

        real_dcg_v = (2 ** relevance - 1) / torch.log2(1 + real_scores_rk_v)
        real_dcg_v = torch.sum(real_dcg_v, dim=-1)

        # NDCG
        real_ndcg_t = real_dcg_t / idcg
        real_ndcg_v = real_dcg_v / idcg

        real_ndcg_t = torch.mean(real_ndcg_t)
        real_ndcg_v = torch.mean(real_ndcg_v)

        ndcg_t = dcg_t / idcg
        ndcg_v = dcg_v / idcg

        loss_t = 1 - ndcg_t
        loss_v = 1 - ndcg_v

        ndcg_t = torch.mean(ndcg_t)
        ndcg_v = torch.mean(ndcg_v)

        loss_t = torch.mean(loss_t)
        loss_v = torch.mean(loss_v)

        loss = (loss_t + loss_v) / 2

        return loss, ndcg_t, real_ndcg_t, ndcg_v, real_ndcg_v


class TripletSNDCGLoss(nn.Module):

    def __init__(self, opt):
        super(TripletSNDCGLoss, self).__init__()
        self.margin = opt.margin
        self.tau = opt.tau
        self.sndcg_weight = opt.sndcg_weight
        self.batch_size = opt.batch_size
        self.pos_mask = torch.eye(self.batch_size).cuda()
        self.neg_mask = 1 - self.pos_mask

    def forward(self, v, t, v_text_emb, t_text_emb):

        batch_size = v.size(0)

        scores = cosine_sim(v, t)
        pos_scores = scores.diag().view(batch_size, 1)
        pos_scores_t = pos_scores.expand_as(scores)
        pos_scores_v = pos_scores.t().expand_as(scores)

        if batch_size != self.batch_size:
            pos_mask = torch.eye(scores.size(0))
            pos_mask = pos_mask.cuda()
            neg_mask = 1 - pos_mask
        else:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        # calculate relevance score
        v_text_emb = v_text_emb.cuda()
        t_text_emb = t_text_emb.cuda()

        v_text_emb = v_text_emb.transpose(0, 1)
        t_text_emb = t_text_emb.view(1, t_text_emb.size(0), t_text_emb.size(1))
        t_text_emb = t_text_emb.expand(5, t_text_emb.size(1), t_text_emb.size(2))

        v_text_emb = l2norm_3d(v_text_emb)
        t_text_emb = l2norm_3d(t_text_emb)
        relevance = torch.bmm(v_text_emb, t_text_emb.transpose(1, 2))
        relevance = relevance.max(0)[0]

        # norm
        relevance = (1 + relevance) / 2  # [0, 1]
        relevance = relevance * neg_mask + pos_mask

        '''pairwise loss'''
        loss_t = (scores - pos_scores_t + self.margin).clamp(min=0)
        loss_v = (scores - pos_scores_v + self.margin).clamp(min=0)
        loss_t = loss_t * neg_mask
        loss_v = loss_v * neg_mask
        loss_t = loss_t.max(dim=1)[0]
        loss_v = loss_v.max(dim=0)[0]
        loss_t = loss_t.mean()
        loss_v = loss_v.mean()
        pairwise_loss = (loss_t + loss_v) / 2

        '''listwise loss'''
        # IDCG
        relevance_repeat = relevance.unsqueeze(dim=2).repeat(1, 1, relevance.size(0))
        relevance_repeat_trans = relevance_repeat.permute(0, 2, 1)
        relevance_diff = relevance_repeat_trans - relevance_repeat
        relevance_indicator = torch.where(relevance_diff > 0,
                                          torch.full_like(relevance_diff, 1),
                                          torch.full_like(relevance_diff, 0))
        relevance_rk = torch.sum(relevance_indicator, dim=-1) + 1
        idcg = (2 ** relevance - 1) / torch.log2(1 + relevance_rk)
        idcg = torch.sum(idcg, dim=-1)

        # scores diff
        scores_repeat_t = scores.unsqueeze(dim=2).repeat(1, 1, scores.size(0))
        scores_repeat_trans_t = scores_repeat_t.permute(0, 2, 1)
        scores_diff_t = scores_repeat_trans_t - scores_repeat_t

        scores_repeat_v = scores.t().unsqueeze(dim=2).repeat(1, 1, scores.size(0))
        scores_repeat_trans_v = scores_repeat_v.permute(0, 2, 1)
        scores_diff_v = scores_repeat_trans_v - scores_repeat_v

        # image-to-text
        scores_sg_t = sigmoid(scores_diff_t, tau=self.tau)

        scores_sg_t = scores_sg_t * neg_mask
        scores_rk_t = torch.sum(scores_sg_t, dim=-1) + 1

        scores_indicator_t = torch.where(scores_diff_t > 0,
                                         torch.full_like(scores_diff_t, 1),
                                         torch.full_like(scores_diff_t, 0))
        real_scores_rk_t = torch.sum(scores_indicator_t, dim=-1) + 1

        dcg_t = (2 ** relevance - 1) / torch.log2(1 + scores_rk_t)
        dcg_t = torch.sum(dcg_t, dim=-1)

        real_dcg_t = (2 ** relevance - 1) / torch.log2(1 + real_scores_rk_t)
        real_dcg_t = torch.sum(real_dcg_t, dim=-1)

        # text-to-image
        scores_sg_v = sigmoid(scores_diff_v, tau=self.tau)

        scores_sg_v = scores_sg_v * neg_mask
        scores_rk_v = torch.sum(scores_sg_v, dim=-1) + 1

        scores_indicator_v = torch.where(scores_diff_v > 0,
                                         torch.full_like(scores_diff_v, 1),
                                         torch.full_like(scores_diff_v, 0))
        real_scores_rk_v = torch.sum(scores_indicator_v, dim=-1) + 1

        dcg_v = (2 ** relevance - 1) / torch.log2(1 + scores_rk_v)
        dcg_v = torch.sum(dcg_v, dim=-1)

        real_dcg_v = (2 ** relevance - 1) / torch.log2(1 + real_scores_rk_v)
        real_dcg_v = torch.sum(real_dcg_v, dim=-1)

        # NDCG
        real_ndcg_t = real_dcg_t / idcg
        real_ndcg_v = real_dcg_v / idcg

        real_ndcg_t = torch.mean(real_ndcg_t)
        real_ndcg_v = torch.mean(real_ndcg_v)

        ndcg_t = dcg_t / idcg
        ndcg_v = dcg_v / idcg

        loss_t = 1 - ndcg_t
        loss_v = 1 - ndcg_v

        ndcg_t = torch.mean(ndcg_t)
        ndcg_v = torch.mean(ndcg_v)

        loss_t = torch.mean(loss_t)
        loss_v = torch.mean(loss_v)

        listwise_loss = (loss_t + loss_v) / 2

        loss = (1 - self.sndcg_weight) * pairwise_loss + self.sndcg_weight * listwise_loss

        return loss, pairwise_loss, listwise_loss, ndcg_t, real_ndcg_t, ndcg_v, real_ndcg_v


class VSE(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt)
        self.txt_enc = EncoderText(opt)

        print(self.img_enc)
        print(self.txt_enc)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.loss = opt.loss
        if self.loss == 'triplet':
            self.criterion = TripletLoss(opt)
        if self.loss == 'sndcg':
            self.criterion = SNDCGLoss(opt)
        if self.loss == 'triplet_sndcg':
            self.criterion = TripletSNDCGLoss(opt)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        images = images.cuda()
        captions = captions.cuda()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, v_text_emb, t_text_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        if self.loss in ['triplet']:
            loss = self.criterion(img_emb, cap_emb)
            self.logger.update('L', loss.item(), img_emb.size(0))
        if self.loss in ['sndcg']:
            loss, ndcg_t, real_ndcg_t, ndcg_v, real_ndcg_v = self.criterion(img_emb, cap_emb, v_text_emb, t_text_emb)
            self.logger.update('L', loss.item(), img_emb.size(0))
            self.logger.update('N1', ndcg_t.item(), img_emb.size(0))
            self.logger.update('RN1', real_ndcg_t.item(), img_emb.size(0))
            self.logger.update('N2', ndcg_v.item(), img_emb.size(0))
            self.logger.update('RN2', real_ndcg_v.item(), img_emb.size(0))
        if self.loss in ['triplet_sndcg']:
            loss, pairwise_loss, listwise_loss, ndcg_t, real_ndcg_t, ndcg_v, real_ndcg_v = \
                self.criterion(img_emb, cap_emb, v_text_emb, t_text_emb)
            self.logger.update('L', loss.item(), img_emb.size(0))
            self.logger.update('PL', pairwise_loss.item(), img_emb.size(0))
            self.logger.update('LL', listwise_loss.item(), img_emb.size(0))
            self.logger.update('N1', ndcg_t.item(), img_emb.size(0))
            self.logger.update('RN1', real_ndcg_t.item(), img_emb.size(0))
            self.logger.update('N2', ndcg_v.item(), img_emb.size(0))
            self.logger.update('RN2', real_ndcg_v.item(), img_emb.size(0))

        return loss

    def train_emb(self, images, captions, lengths, image_ids, caption_ids, v_text_emb, t_text_emb, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, v_text_emb, t_text_emb)

        # compute gradient and do SGD step
        loss.backward()
        clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
