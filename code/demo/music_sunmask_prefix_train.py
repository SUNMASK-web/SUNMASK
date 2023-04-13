import re
import gzip
import csv
from collections import Counter, OrderedDict
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from music_sunmask_prefix_model import PerceiverSeq
from music_sunmask_prefix_model import RampOpt
from music_sunmask_prefix_model import clipping_grad_norm_
from music_sunmask_prefix_model import clipping_grad_value_
from music_sunmask_prefix_model import get_device_default, set_device_default

# data loading
# dataset precalculated from https://github.com/kastnerkyle/kkpthlib/blob/basics/examples/scripts/make_simple_music_dataset.py
d = np.load('Jsb16thSeparatedAlignedFull.npz', encoding='bytes', allow_pickle=True)
train_data = d["train_data_full"]
test_data = d["test_data_full"]
train_data_attribution = d["train_data_attribution_full"]
test_data_attribution = d["test_data_attribution_full"]

max_midi_pitch = -1
min_midi_pitch = np.inf
all_tracks = []
all_tracks_attribution = []
for _i in range(len(train_data)):
    #if "transposed" not in train_data_attribution[_i]:
        all_tracks.append(train_data[_i].astype("int32"))
        all_tracks_attribution.append(train_data_attribution[_i])
        this_ = all_tracks[-1].ravel()
        this_min = this_[this_ > 0].min()
        this_max = this_[this_ > 0].max()
        if this_min < min_midi_pitch:
            min_midi_pitch = this_min
        if this_max > max_midi_pitch:
            max_midi_pitch = this_max

holdout_size = int(.05 * len(all_tracks))
valid_rng = np.random.RandomState(9876)
valid_inds = valid_rng.choice(np.arange(len(all_tracks)), size=holdout_size, replace=False)
train_tracks = [t for _n, t in enumerate(all_tracks) if _n not in valid_inds]
train_tracks_attribution = [t for _n, t in enumerate(all_tracks_attribution) if _n not in valid_inds]

valid_tracks = [t for _n, t in enumerate(all_tracks) if _n in valid_inds]
valid_tracks_attribution = [t for _n, t in enumerate(all_tracks_attribution) if _n in valid_inds]

test_tracks = []
test_tracks_attribution = []
for _i in range(len(test_data)):
    #if "transposed" not in test_data_attribution[_i]:
        test_tracks.append(test_data[_i].astype("int32"))
        test_tracks_attribution.append(test_data_attribution[_i])
        this_ = test_tracks[-1].ravel()
        this_min = this_[this_ > 0].min()
        this_max = this_[this_ > 0].max()
        if this_min < min_midi_pitch:
            min_midi_pitch = this_min
        if this_max > max_midi_pitch:
            max_midi_pitch = this_max

print("Number train seq", len(train_tracks))
print("Number valid seq", len(valid_tracks))
print("Number test seq ", len(test_tracks))
print("Min midi pitch", min_midi_pitch)
print("Max midi pitch", max_midi_pitch)

prefix_length = 256
# this is actually in total steps, so effectively // num_instruments aka 32
latent_length = 256
n_unrolled_steps = 2

working_batch_size = 15
batch_size = 30
assert batch_size / float(working_batch_size) == batch_size // working_batch_size
batch_aggregations = batch_size // working_batch_size

total_length = prefix_length + latent_length

num_instruments = 4
# make the datasets now
# do every single measure with proper prior context
# for now assume we start with seeded gen, can work out unseeded later
assert latent_length / float(num_instruments) == latent_length // num_instruments
assert total_length / float(num_instruments) == total_length // num_instruments
target_length_meas = latent_length // num_instruments
total_length_meas = total_length // num_instruments
scoot = 16 # 1 is 1 step for all 4 voices
cut_train_tracks = []
cut_train_tracks_attribution = []
for _i in range(len(train_tracks)):
    pos = 0
    span = total_length_meas
    this_track = train_tracks[_i]
    this_track_attribution = train_tracks_attribution[_i]
    while True:
        if (pos + span) > this_track.shape[1]:
            break
        else:
            c = this_track[:, pos:pos + span]
            # skip things with rests
            if c.min() > 0:
                cut_train_tracks.append(c)
                cut_train_tracks_attribution.append((this_track_attribution, (pos, pos+span)))
        # scoot 1 measure at a time
        pos += scoot

cut_valid_tracks = []
cut_valid_tracks_attribution = []
for _i in range(len(valid_tracks)):
    pos = 0
    span = total_length_meas
    this_track = valid_tracks[_i]
    this_track_attribution = valid_tracks_attribution[_i]
    while True:
        if (pos + span) > this_track.shape[1]:
            break
        else:
            c = this_track[:, pos:pos + span]
            if c.min() > 0:
                cut_valid_tracks.append(c)
                cut_valid_tracks_attribution.append((this_track_attribution, (pos, pos+span)))
        # scoot 1 measure at a time
        pos += scoot
cut_train_tracks = np.array(cut_train_tracks).astype("int32") - min_midi_pitch
cut_valid_tracks = np.array(cut_valid_tracks).astype("int32") - min_midi_pitch
# data processing and normalization complete

class Net(nn.Module):
    def __init__(self, train_mode=True):
        super().__init__()
        # internal plus 1 to allow the max and mix val
        self.n_classes = ((max_midi_pitch - min_midi_pitch) + 1) + 1
        self.latent_length = latent_length
        self.n_layers = 16
        self.hidden_size = 380
        self.self_inner_dim = 900
        if train_mode:
            self.input_dropout_keep_prob = 1.0
            # cross attend keep prob for uniform functions as apply prob?
            # probability to not apply the mask is 1 - keep_prob
            # low keep prob -> mostly do not apply mask
            # 1.0 -> always apply uniform mask (.5 expectation)
            # when set to "default", keep prob functions as expected
            # 1.0 -> all keep
            # 0.5 -> 50% dropout
            self.cross_attend_dropout_keep_prob = 0.5 #1.0
            self.cross_attend_dropout_type = "uniform" #"default"
            self.autoregression_dropout_keep_prob = 1.0
            self.autoregression_dropout_type = "uniform" #"default"
            # inner and final from transformer xl
            self.inner_dropout_keep_prob = .8
            self.final_dropout_keep_prob = 1.0
        else:
            self.input_dropout_keep_prob = 1.0
            self.cross_attend_dropout_keep_prob = 1.0
            self.cross_attend_dropout_type = "default"
            self.autoregression_dropout_keep_prob = 1.0 #1.0
            self.autoregression_dropout_type = "default"
            # inner and final from transofmer xl
            self.inner_dropout_keep_prob = 1.0
            self.final_dropout_keep_prob = 1.0
        self.learnable_position_embeddings = True
        self.tied_embeddings = True
        self.position_encoding_type = "rotary"
        self.fraction_to_rotate = 0.25
        self.fraction_heads_to_rotate = 1.0
        self.cross_attn_heads = 1
        self.self_attn_heads = 10
        """
        hidden_size = 380
        self_inner_dim = 900
        input_dropout_keep_prob = 0.8
        cross_attend_dropout_keep_prob = 0.25
        autoregression_dropout_keep_prob = 1.0
        inner_dropout_keep_prob = 1.0
        final_dropout_keep_prob = 0.5
        n_layers = 20
        """
        self.perceiver = PerceiverSeq(autoregressive_mode=False,
                                     n_classes=self.n_classes,
                                     z_index_dim=self.latent_length,
                                     n_processor_layers=self.n_layers,
                                     input_embed_dim=self.hidden_size,
                                     num_z_channels=self.hidden_size,
                                     inner_expansion_dim=self.self_inner_dim,
                                     input_dropout_keep_prob=self.input_dropout_keep_prob,
                                     cross_attend_dropout_keep_prob=self.cross_attend_dropout_keep_prob,
                                     cross_attend_dropout_type=self.cross_attend_dropout_type,
                                     autoregression_dropout_keep_prob=self.autoregression_dropout_keep_prob,
                                     autoregression_dropout_type=self.autoregression_dropout_type,
                                     inner_dropout_keep_prob=self.inner_dropout_keep_prob,
                                     final_dropout_keep_prob=self.final_dropout_keep_prob,
                                     cross_attn_heads=self.cross_attn_heads,
                                     self_attn_heads=self.self_attn_heads,
                                     learnable_position_embeddings=self.learnable_position_embeddings,
                                     tied_embeddings=self.tied_embeddings,
                                     position_encoding_type=self.position_encoding_type,
                                     fraction_to_rotate=self.fraction_to_rotate,
                                     fraction_heads_to_rotate=self.fraction_heads_to_rotate)

    def forward(self, inputs, input_idxs, input_mask=None):
        # mask None means do it autoregressively
        logits, input_masks, drop_masks, base_drop_masks = self.perceiver(inputs, input_idxs, input_mask)
        return logits, drop_masks

data_loader_random_state = np.random.RandomState(2123)
def load_minibatch(batch_size, split="train", fixed_index=None):
    train_inds = range(len(cut_train_tracks))
    valid_inds = range(len(cut_valid_tracks))
    train_data = cut_train_tracks
    valid_data = cut_valid_tracks
    batch_indices = train_inds if split == "train" else valid_inds
    data_set = train_data if split == "train" else valid_data

    if fixed_index is not None:
        sampled_batch_indices = [fixed_index for _ in range(minibatch_size)]
    else:
        sampled_batch_indices = data_loader_random_state.choice(batch_indices, size=batch_size, replace=True)

    batch_sents = []
    for _bi in sampled_batch_indices:
        sent = data_set[_bi]
        batch_sents.append(sent)
    return np.array(batch_sents)

def make_batch(batch):
    batch = batch + 1
    shp = batch.shape
    # reshaping B, I, T -> B, I * T will order it SSSSAAAAATTTTTBBBB
    # want interleaved
    sequence_length = shp[1] * shp[2]
    batch = batch.transpose(0, 2, 1).reshape((shp[0], sequence_length))
    # generate corresponding idx, we assume all entries "fill" measure, no 0 padding
    batch_idx = 0. * batch + np.arange(sequence_length)[None]
    # batch now has correct shape overall, and is interleaved
    # swap to T, B format
    batch = batch.transpose(1, 0)
    batch_idx = batch_idx.transpose(1, 0).astype("int32")
    # idx has trailing 1
    batch_idx = batch_idx[..., None]
    # was 0 min, now 1 min (0 for padding in future datasets)
    targets = copy.deepcopy(batch[-latent_length:])
    # 0 is a special token, so batch is 1:n_classes + 1
    # but targets could be moved to 0:n_classes?
    return batch, batch_idx, targets

#data_mb = load_minibatch(batch_size, "train")
#data_batch, batch_idx, target_batch = make_batch(data_mb)

n_train_steps = 100_000#50_000
learning_rate = 0.0003
clip_grad = 3
min_learning_rate = 3E-6
ramp_til = 5000
decay_til = n_train_steps - 10000 #5000
save_every = int(n_train_steps / 20)
valid_every = 50

model_save_path = "music_sunmask_prefix_models"
model_fpath = os.path.join(model_save_path, 'music_sunmask_prefix.pth')
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)


if __name__ == "__main__":
    #set_device_default("cuda")
    device = get_device_default()
    model = Net().to(device)
    mask_random_state = np.random.RandomState(2234)
    data_random_state = np.random.RandomState(2122)
    gumbel_sampling_random_state = np.random.RandomState(3434)
    corruption_sampling_random_state = np.random.RandomState(1122)

    def get_std_ramp_opt(model):
        return RampOpt(learning_rate, 1, ramp_til, decay_til,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.1, 0.999), eps=1E-6),
                       min_decay_learning_rate=min_learning_rate)
    optimizer = get_std_ramp_opt(model)

    # speed this up with torch generator?
    def gumbel_sample(logits, temperature=1., low=0):
        noise = gumbel_sampling_random_state.uniform(1E-5, 1. - 1E-5, logits.shape)
        torch_noise = torch.tensor(noise).contiguous().to(device)

        #return np.argmax(np.log(softmax(logits, temperature)) - np.log(-np.log(noise)))
        # max indices
        # no keepdim here
        if low != 0:
            maxes = torch.argmax(logits[..., low:] / float(temperature) - torch.log(-torch.log(torch_noise[..., low:])), axis=-1)
            maxes = maxes + low
        else:
            maxes = torch.argmax(logits / float(temperature) - torch.log(-torch.log(torch_noise)), axis=-1)
        return maxes

    # same here - torch generator speed it up?
    def get_random_pitches(shape, vocab_size, low=0):
        # add 1 due to batch offset reserving 0 for length masking
        r = corruption_sampling_random_state.randint(low=low, high=vocab_size, size=shape)
        random_pitch = torch.tensor(copy.deepcopy(r)).type(torch.LongTensor).to(device)
        return random_pitch

    def corrupt_pitch_mask(batch, mask, vocab_size, low=0):
        random_pitches = get_random_pitches(batch.shape, vocab_size, low=low)
        #corrupted = (1 - mask[..., None]) * random_pitches + (1 * mask[..., None]) * batch
        corrupted = (1 - mask) * random_pitches + (1 * mask) * batch
        return corrupted

    # SUNDAE https://arxiv.org/pdf/2112.06749.pdf
    def build_logits_fn(vocab_size, n_unrolled_steps, enable_sampling):
        def logits_fn(input_batch, input_batch_idx, input_mask):
            def fn(batch, batch_idx, mask):
                logits, masks = model(batch, batch_idx, mask)
                return logits

            def unroll_fn(batch, batch_idx, mask):
                # only corrupt query ones - this will break for uneven seq lengths!
                # vocab_size -1 , low = 1 in corrupt and gumbel to handle perceiver 0 issue
                samples = corrupt_pitch_mask(batch[-latent_length:], mask, vocab_size - 1, low=1)
                samples = torch.concat([batch[:-latent_length], samples], axis=0)
                all_logits = []
                for _ in range(n_unrolled_steps):
                    logits = fn(samples, batch_idx, mask)
                    samples = gumbel_sample(logits, low=1).detach()
                    # sanity check to avoid issues with stacked outputs
                    assert samples.shape[1] == batch.shape[1]
                    # for the SUNDAE piece
                    samples = samples[:, :batch.shape[1]]
                    samples = torch.concat([batch[:-latent_length], samples], axis=0)
                    all_logits += [logits[None]]
                final_logits = torch.cat(all_logits, dim=0)
                return final_logits

            if enable_sampling:
                return fn(input_batch, input_batch_idx, input_mask)
            else:
                return unroll_fn(input_batch, input_batch_idx, input_mask)
        return logits_fn

    def build_loss_fn(vocab_size, n_unrolled_steps=4):
        logits_fn = build_logits_fn(vocab_size, n_unrolled_steps, enable_sampling=False)

        def local_loss_fn(batch, batch_idx, mask, targets):
            # repeated targets are now n_unrolled_steps
            repeated_targets = torch.cat([targets[..., None]] * n_unrolled_steps, dim=1)
            # T N 1 -> N T 1
            repeated_targets = repeated_targets.permute(1, 0, 2)
            assert repeated_targets.shape[-1] == 1
            lcl_batch_size = repeated_targets.shape[0]
            # N T 1 -> N T P

            repeated_targets = F.one_hot(repeated_targets[..., 0].long(), num_classes=vocab_size)
            #t = torch.argmax(repeated_targets, axis=-1)
            #for i in range(t.shape[0]):
            #    print([ind_to_vocab[int(e)] for e in t[i].cpu().data.numpy()])
            #print(mask)

            logits = logits_fn(batch, batch_idx, mask)
            # S, T, N, P -> S, N, T, P
            logits = logits.permute(0, 2, 1, 3)
            out = logits.reshape(n_unrolled_steps * logits.shape[1], logits.shape[2], logits.shape[3])
            logits = out
            # N, T, P
            #? trouble
            raw_loss = -1. * (nn.functional.log_softmax(logits, dim=-1) * repeated_targets)
            # mask is currently T, N
            # change to N, T, 1, then stack for masking
            # only keep loss over positions which were dropped, no freebies here
            raw_masked_loss = raw_loss * torch.cat([(1. - mask.permute(1, 0)[..., None])] * n_unrolled_steps, dim=0)
            raw_unmasked_loss = raw_loss * torch.cat([(mask.permute(1, 0)[..., None])] * n_unrolled_steps, dim=0)
            reduced_mask_active = torch.cat([1. / ((1. - mask).sum(dim=0) + 1)] * n_unrolled_steps, dim=0)[..., None, None]
            # masked entries are the target
            raw_comb_loss = raw_masked_loss
            # Active mask sums up the amount that were inactive in time
            # downweighting more if more were not dropped out
            reduced_loss = (reduced_mask_active * raw_comb_loss).sum(dim=-1)
            loss = torch.mean(reduced_loss, dim=1).mean()
            return loss
        return local_loss_fn

    u_loss_fn = build_loss_fn(model.n_classes, n_unrolled_steps=n_unrolled_steps)
    np_train_losses = [(0, -1)]
    np_valid_losses = [(0, -1)]
    for _n in range(int(n_train_steps)):
        optimizer.zero_grad()
        data_mb = load_minibatch(batch_size, "train")
        loss_acc = 0.
        raw_loss_acc = 0.
        for _bs in range(batch_aggregations):
            start_point = working_batch_size * (_bs)
            end_point = working_batch_size * (_bs + 1)
            data_batch, batch_idx, target_batch = make_batch(data_mb[start_point:end_point])

            # mask is only latent_length long
            C_prob = mask_random_state.rand(data_batch.shape[1])
            C_mask_base = mask_random_state.rand(data_batch[-latent_length:].shape[0], data_batch.shape[1])
            C = 1 * (C_mask_base < C_prob[None, :])
            C = (1. - C) # convert to 0 drop format
            C = C.astype(np.int32)

            x = torch.tensor(data_batch).type(torch.FloatTensor).to(device)
            x_idx = torch.tensor(batch_idx).type(torch.FloatTensor).to(device)
            targets = torch.tensor(target_batch).type(torch.FloatTensor).to(device)
            mask = torch.tensor(C).type(torch.FloatTensor).to(device)
            # skip the mask for now
            #mask = None

            #logits, drop_masks = model(x, x_idx, mask)
            raw_loss = u_loss_fn(x, x_idx, mask, targets)
            loss = raw_loss / batch_aggregations
            loss_acc += loss.cpu().data.numpy()
            loss.backward()

        l = loss_acc
        if np_train_losses[-1][-1] == -1:
            np_train_losses[0] = (_n, l)
        else:
            np_train_losses.append((_n, l))
        clipping_grad_value_(model.parameters(), clip_grad)
        #clipping_grad_value_(model.named_parameters(), clip_grad, named_check=True)
        optimizer.step()

        if _n % 200 == 0:
            print('train step {}, train_loss: {:.6f}, train_pseudoppl {:.6f}'.format(
                _n, np_train_losses[-1][1], np.exp(np_train_losses[-1][-1])))
            print('valid step {}, valid_loss: {:.6f}, valid_pseudoppl {:.6f}'.format(
                _n, np_valid_losses[-1][1], np.exp(np_valid_losses[-1][-1])))
        if _n != 0 and (_n % save_every == 0 or _n == (n_train_steps - 1)):
            model_fpath_base = model_fpath.split(".pth")[0]
            model_fpath_iter = model_fpath_base + "_{}".format(_n) + ".pth"
            torch.save(model.state_dict(), model_fpath_iter)
            model_fpath_metrics = model_fpath_base + "_metrics.npz"
            np.savez(model_fpath_metrics,
                     train_losses=np.array(np_train_losses),
                     valid_losses=np.array(np_valid_losses))

        if _n > 1 and _n % valid_every == 0:
            optimizer.zero_grad()
            with torch.no_grad():
                data_mb = load_minibatch(batch_size, "valid")
                loss_acc = 0.
                for _bs in range(batch_aggregations):
                    start_point = working_batch_size * (_bs)
                    end_point = working_batch_size * (_bs + 1)
                    data_batch, batch_idx, target_batch = make_batch(data_mb[start_point:end_point])

                    # mask is only latent_length long
                    C_prob = mask_random_state.rand(data_batch.shape[1])
                    C_mask_base = mask_random_state.rand(data_batch[-latent_length:].shape[0], data_batch.shape[1])
                    C = 1 * (C_mask_base < C_prob[None, :])
                    C = (1. - C) # convert to 0 drop format
                    C = C.astype(np.int32)

                    x = torch.tensor(data_batch).type(torch.FloatTensor).to(device)
                    x_idx = torch.tensor(batch_idx).type(torch.FloatTensor).to(device)
                    targets = torch.tensor(target_batch).type(torch.FloatTensor).to(device)
                    mask = torch.tensor(C).type(torch.FloatTensor).to(device)
                    # skip the mask for now
                    #mask = None

                    raw_loss = u_loss_fn(x, x_idx, mask, targets)
                    loss = raw_loss / batch_aggregations
                    loss_acc += loss.cpu().data.numpy()
                l = loss_acc
                if np_valid_losses[-1] == -1:
                    np_valid_losses[0] = (_n, l)
                else:
                    np_valid_losses.append((_n, l))
            optimizer.zero_grad()
    from IPython import embed; embed(); raise ValueError()
