import re
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
import torch
import numpy as np
import torch.nn as nn
import mido
import pretty_midi
import copy
import os
import gzip
import csv
from collections import OrderedDict

class Chorale:
    """
    A class to store and manipulate an array self.arr that stores a chorale.
    Arr is an array of shape (4, num_timesteps) with values in range(0, vocab_size)
    """
    def __init__(self, arr,
                 vocab_size,
                 num_instruments=4):
        # arr is an array of shape (4, 32) with values in range(0, 57)
        self.arr = arr.copy()

        assert num_instruments == arr.shape[0]

        # the one_hot representation of the array
        reshaped = self.arr.reshape(-1)
        self.num_timesteps = arr.shape[-1]
        self.num_instruments = num_instruments
        self.vocab_size = vocab_size
        self.one_hot = np.zeros((num_instruments * self.num_timesteps, vocab_size))

        r = np.arange(num_instruments * self.num_timesteps)
        self.one_hot[r, reshaped] = 1
        self.one_hot = self.one_hot.reshape(num_instruments, self.num_timesteps, vocab_size)

    def piano_roll_to_midi(self, piece, program_map_satb=[52, 52, 52, 52], velocity_map_satb=[70, 70, 70, 70], bpm=50):
        """
        Function for converting arrays of shape (T, 4) into midi files
        the input array has entries that are np.nan (representing a rest)
        of an integer between 0 and 127 inclusive

        piece is a an array of shape (T, 4) for some T.
        The (i,j)th entry of the array is the midi pitch of the jth voice at time i. It's an integer in range(128).
        outputs a mido object mid that you can convert to a midi file by called its .save() method
        """
        piece = np.concatenate([piece, [[np.nan, np.nan, np.nan, np.nan]]], axis=0)

        microseconds_per_beat = 60 * 1000000 / bpm

        mid = mido.MidiFile()
        tracks = OrderedDict()
        tracks["soprano"] = mido.MidiTrack()
        tracks["alto"] = mido.MidiTrack()
        tracks["tenor"] = mido.MidiTrack()
        tracks["bass"] = mido.MidiTrack()

        past_pitches = {'soprano': np.nan, 'alto': np.nan,
                        'tenor': np.nan, 'bass': np.nan}
        delta_time = {'soprano': 0, 'alto': 0, 'tenor': 0, 'bass': 0}

        # create a track containing tempo data
        metatrack = mido.MidiTrack()
        metatrack.append(mido.MetaMessage('set_tempo', tempo=int(microseconds_per_beat), time=0))
        mid.tracks.append(metatrack)

        # create the four voice tracks
        for _i, voice in enumerate(tracks):
            mid.tracks.append(tracks[voice])
            tracks[voice].append(mido.Message(
                'program_change', program=program_map_satb[_i], time=0))

        # add notes to the four voice tracks
        for i in range(len(piece)):
            pitches = {'soprano': piece[i, 0], 'alto': piece[i, 1], 'tenor': piece[i, 2], 'bass': piece[i, 3]}
            for _j, voice in enumerate(tracks):
                if np.isnan(past_pitches[voice]):
                    past_pitches[voice] = None
                if np.isnan(pitches[voice]):
                    pitches[voice] = None
                if pitches[voice] != past_pitches[voice]:
                    if past_pitches[voice]:
                        tracks[voice].append(mido.Message('note_off', note=int(past_pitches[voice]),
                                             velocity=velocity_map_satb[_j], time=delta_time[voice]))
                        delta_time[voice] = 0
                    if pitches[voice]:
                        tracks[voice].append(mido.Message('note_on', note=int(pitches[voice]),
                                             velocity=velocity_map_satb[_j], time=delta_time[voice]))
                        delta_time[voice] = 0
                past_pitches[voice] = pitches[voice]
                # 480 ticks per beat and each line of the array is a 16th note
                delta_time[voice] += 120
        return mid

    def save(self, filename="download.mid",
             program_map_satb=[52, 52, 52, 52],
             velocity_map_satb=[70, 70, 70, 70],
             bpm=50,
             shift_midi_by_offset=27):
        midi_arr = self.arr.transpose().copy()
        # shift?
        midi_arr += shift_midi_by_offset
        midi = self.piano_roll_to_midi(midi_arr, program_map_satb, velocity_map_satb, bpm)
        midi.save(filename)

    def to_image(self, shift_midi_by_offset=27):
        plt.style.use("seaborn-v0_8-ticks") #'light_background')

        # visualize the four tracks as individual images
        soprano_oh = self.one_hot[0]
        alto_oh = self.one_hot[1]
        tenor_oh = self.one_hot[2]
        bass_oh = self.one_hot[3]

        _o = shift_midi_by_offset
        soprano_m = soprano_oh.argmax(axis=-1) + _o
        alto_m = alto_oh.argmax(axis=-1) + _o
        tenor_m = tenor_oh.argmax(axis=-1) + _o
        bass_m = bass_oh.argmax(axis=-1) + _o

        soprano = np.zeros((self.num_timesteps, self.vocab_size + _o))
        alto = np.zeros((self.num_timesteps, self.vocab_size + _o))
        tenor = np.zeros((self.num_timesteps, self.vocab_size + _o))
        bass = np.zeros((self.num_timesteps, self.vocab_size + _o))

        r = np.arange(self.num_timesteps)
        soprano[r, soprano_m] = 1
        #soprano = soprano[:, 30:]
        alto[r, alto_m] = 1
        #alto = alto[:, 30:]
        tenor[r, tenor_m] = 1
        #tenor = tenor[:, 30:]
        bass[r, bass_m] = 1
        #bass = bass[:, 30:]

        soprano = soprano.transpose()
        alto = alto.transpose()
        tenor = tenor.transpose()
        bass = bass.transpose()

        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(soprano, cmap='Greys', interpolation='nearest')
        axs[0].set_title('soprano')
        axs[0].invert_yaxis()
        #axs[0].set_xticks([])
        axs[0].set_ylim(ymax=self.vocab_size + _o, ymin=_o)
        axs[0].set_xlim(xmax=self.num_timesteps, xmin=0)

        axs[1].imshow(alto, cmap='Greys', interpolation='nearest')
        axs[1].set_title('alto')
        axs[1].invert_yaxis()
        #axs[1].set_xticks([])
        axs[1].set_ylim(ymax=self.vocab_size + _o, ymin=_o)
        axs[1].set_xlim(xmax=self.num_timesteps, xmin=0)


        axs[2].imshow(tenor, cmap='Greys', interpolation='nearest')
        axs[2].set_title('tenor')
        axs[2].invert_yaxis()
        #axs[2].set_xticks([])
        axs[2].set_ylim(ymax=self.vocab_size + _o, ymin=_o)
        axs[2].set_xlim(xmax=self.num_timesteps, xmin=0)


        axs[3].imshow(bass, cmap='Greys', interpolation='nearest')
        axs[3].set_title('bass')
        axs[3].invert_yaxis()
        #axs[3].set_xticks([])
        axs[2].set_ylim(ymax=self.vocab_size + _o, ymin=_o)
        axs[2].set_xlim(xmax=self.num_timesteps, xmin=0)

        fig.set_figheight(5)
        fig.set_figwidth(15)
        return fig, axs

    def to_image_combined(self, shift_midi_by_offset=27):
        chorale = self
        _o = shift_midi_by_offset
        midi = self.piano_roll_to_midi(chorale.arr.transpose().copy() + _o)
        base_dir = os.getcwd()
        if not os.path.exists(base_dir + os.sep + 'midi_files'):
            os.mkdir(base_dir + os.sep + 'midi_files' + os.sep)
        midi.save(base_dir + os.sep + 'midi_files' + os.sep + 'tmp_for_plot.mid')
        midi_data = pretty_midi.PrettyMIDI(base_dir + os.sep + 'midi_files' + os.sep + 'tmp_for_plot.mid')
        total_length = midi_data.get_end_time()
        plt.style.use("seaborn-v0_8-ticks") #'light_background')
        #plt.style.use('dark_background')
        fig = plt.figure()
        ax = plt.gca()
        ax.set_ylim(ymax=self.vocab_size + _o, ymin=_o)
        #ax.set_xlim(xmax=total_length, xmin=0)
        #ax.set_xlim(xmax=T, xmin=0)
        colors = [(189, 39, 25), #SATB
                  (36, 88, 197),
                  (228, 174, 74),
                  (121, 165, 90),
                ]

        colors = [tuple([c/255 for c in color]) for color in colors]

        part_names = ['soprano', 'alto', 'tenor', 'bass']
        alpha = 0.8

        for i in range(4):
            note_list = midi_data.instruments[i].notes
            color = colors[i]

            for j in range(len(note_list)):
                note = note_list[j]
                pitch = note.pitch
                on = note.start
                off = note.end
                if j==len(note_list)-1:
                    ax.axhline(y=pitch, xmin=(on/total_length), xmax=(off/total_length),
                               lw=4, color=color, alpha=alpha, solid_capstyle='butt', label=part_names[i])
                else:
                    ax.axhline(y=pitch, xmin=(on/total_length), xmax=(off/total_length),
                               lw=4, color=color, alpha=alpha, solid_capstyle='butt')
        ax.set_ylabel("Pitch")
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=5)
        ax.set_xticks(np.arange(0, 5+1) * 25)
        return fig, ax

def np_logsumexp(x, axis=-1):
    if axis != -1:
        raise ValueError("Axis must be -1")
    mx = np.max(x, axis=axis, keepdims=True)
    s = x - mx
    s_exp = np.exp(s).sum(axis=axis)
    return mx[..., 0] + np.log(s_exp)

def np_softmax(x, axis=-1):
    if axis != -1:
        raise ValueError("Axis must be -1")
    em_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return em_x / (em_x.sum(axis=-1, keepdims=True) + (em_x.sum(axis=-1) == 0)[..., None])

from music_sunmask_prefix_model import get_device_default
from music_sunmask_prefix_model import typical_top_k_filtering
from music_sunmask_prefix_model import top_k_top_p_filtering
from music_sunmask_prefix_train import *

import sys
if len(sys.argv) < 2:
    raise ValueError("Required argument should specify the saved model weights to load!")
saved_model_path = sys.argv[1]
if len(sys.argv) == 3:
   saved_metrics_path = sys.argv[2]
else:
   saved_metrics_path = None

device = get_device_default()
model = Net(train_mode=False).to(device)
print("Loading model from {}".format(saved_model_path))
model.load_state_dict(torch.load(saved_model_path, map_location=device))

if saved_metrics_path is not None:
    metrics = np.load(saved_metrics_path)
    metrics_keys = sorted(list(metrics.keys()))
    plot_keys = ["train_losses", "valid_losses"]
    for mk in metrics_keys:
        if mk in plot_keys:
            if mk == "valid_losses":
                steps = metrics[mk][1:, 0]
                vals = metrics[mk][1:, 1]
            else:
                steps = metrics[mk][:, 0]
                vals = metrics[mk][:, 1]
            plt.plot(steps, vals, label=mk)
            if mk == "train_losses":
                plt.xlabel("last 1k step train mean: {:.4f}".format(np.mean(vals[-1000:])))
    plt.legend()
    plt.savefig("train_metrics.png")

# harmonize a melody
def torch_harmonize_perceiversunmask(y, C, model,
                                     n_steps, # I * T default
                                     batch_fn,
                                     vocab_size,
                                     internal_batch_size=2,
                                     keep_mask=None,
                                     n_reps_per_mask=1,
                                     n_reps_final_mask_dwell=0,
                                     sundae_keep_prob=0.33,
                                     initial_corrupt=True,
                                     intermediate_corrupt=False,
                                     frozen_mask=False,
                                     use_evener=False,
                                     top_k=0, top_p=0.0,
                                     swap_at_eta=False,
                                     use_typical_sampling=False,
                                     temperature=1.0, o_nade_eta=3./4, seed=12,
                                     return_intermediates=False,
                                     verbose=True):
    """
    Generate an artificial Bach Chorale starting with unfolded batch y (N, 4, T)
    and keeping the pitches where C==1.
    Here C is an array of shape (N, 4, latent length) whose entries are 0 and 1.
    The pitches outside of C are repeatedly resampled to generate new values.
    For example, to harmonize the soprano line, let y be random except y[0] contains the soprano line, let C[1:] be 0 and C[0] be 1.
    """
    lcl_seed_track = copy.deepcopy(y)
    if len(lcl_seed_track.shape) == 2:
        lcl_seed_track = np.concatenate([lcl_seed_track[None] for _ in range(batch_size)])
        C = np.concatenate([C[None] for _ in range(batch_size)])
    batch, batch_idx, _ = batch_fn(lcl_seed_track)
    n_samples = batch.shape[1]

    C = C.astype(np.int32)
    # C is batch, I, latent_length//I
    # flatten to batch, I * latent_length//I
    # then switch to I * latent_length//I, batch
    # reshaping B, I, T -> B, I * T will order it SSSSAAAAATTTTTBBBB
    # want interleaved
    shp = C.shape
    C = C.transpose(0, 2, 1).reshape(shp[0], shp[1] * shp[2])
    # latent_length, B
    C = C.transpose(1, 0)

    batch = batch[:, :internal_batch_size]
    batch_idx = batch_idx[:, :internal_batch_size]
    C = C[:, :internal_batch_size]

    x = torch.tensor(batch).type(torch.FloatTensor).to(device)
    x_idx = torch.tensor(batch_idx).type(torch.FloatTensor).to(device)
    # due to sampling interface we need to flatten the C mask to SATBSATBSATB order...
    C = torch.tensor(C).long().to(device)

    model.eval()
    rs = np.random.RandomState(seed)
    trsg = torch.Generator(device=device)
    trsg.manual_seed(seed)

    def lcl_gumbel_sample(logits, temperature=1., low=0):
        #noise = rs.uniform(1E-5, 1. - 1E-5, logits.shape)
        #torch_noise = torch.tensor(noise).contiguous().to(device)
        torch_noise = torch.rand(logits.shape, generator=trsg, device=device) * ((1 - 1E-5) - 1E-5) + 1E-5

        #return np.argmax(np.log(softmax(logits, temperature)) - np.log(-np.log(noise)))

        # max indices
        #maxes = torch.argmax(logits / float(temperature) - torch.log(-torch.log(torch_noise)), axis=-1, keepdim=True)
        if low != 0:
            maxes = torch.argmax(logits[..., low:] / float(temperature) - torch.log(-torch.log(torch_noise[..., low:])), axis=-1)
            maxes = maxes + low
        else:
            maxes = torch.argmax(logits / float(temperature) - torch.log(-torch.log(torch_noise)), axis=-1)
        return maxes
        #one_hot = 0. * logits
        #one_hot.scatter_(-1, maxes, 1)
        #return one_hot

    def lcl_get_random_pitches(shape, vocab_size, low=0):
        random_pitch = torch.randint(low=low, high=vocab_size, size=shape, device=device, generator=trsg)
        return random_pitch

    with torch.no_grad():
        if keep_mask is not None:
            keep_C = torch.tensor(keep_mask).long().to(device)

        C2 = torch.clone(C)#.copy()
        #num_steps = int(2*I*T)
        alpha_max = .999
        alpha_min = .001
        eta = o_nade_eta

        """
        for i in range(num_steps):
            p = np.maximum(alpha_min, alpha_max - i*(alpha_max-alpha_min)/(eta*num_steps))
            sampled_binaries = rs.choice(2, size = C.shape, p=[p, 1-p])
            C2 += sampled_binaries
            C2[C==1] = 1
            x_cache = x
            x = model.pred(x, C2, seed=rs.randint(100000))
            x[C2==1] = x_cache[C2==1]
            C2 = C.copy()
        """

        x_cache = torch.clone(x)

        if initial_corrupt:
            x_sub = lcl_get_random_pitches(x[-latent_length:].shape, vocab_size - 1, low=1).float()
            # add 1 since 0 is protected value for masking
            x[-latent_length:] = x_sub
            x[-latent_length:][C2==1] = x_cache[-latent_length:][C2 == 1]
            if keep_mask is not None:
                x[-latent_length:][keep_C==1] = x_cache[keep_C==1]
        # x is now corrupted in the portion corresponding to the query

        n_steps = max(1, int(n_steps))
        if sundae_keep_prob == "triangular":
            sundae_keep_tokens_per_step = [2 * C.shape[0] * min((t + 1) / float(n_steps), 1 - (t + 1) / float(n_steps))
                                           for t in range(int(n_steps))] + [1.0 * C.shape[0] for t in range(int(n_reps_final_mask_dwell))]
        else:
            sundae_keep_tokens_per_step = [sundae_keep_prob * C.shape[0]
                                          for t in range(int(n_steps))] + [1.0 * C.shape[0] for t in range(int(n_reps_final_mask_dwell))]

        has_been_kept = 1. + 0. * C
        has_been_kept_torch = has_been_kept.clone().detach().to(device)

        # might need to renormalize the kept matrix at some point...
        sampled_binaries = None
        all_x = []
        all_C = []
        for n in range(int(n_steps + n_reps_final_mask_dwell)):
            try:
                step_top_k = top_k[0]
                min_k = min(top_k)
                max_k = max(top_k)
                step_frac_k = (max_k - min_k) / float(n_steps)
                if min_k == top_k[1]:
                    step_top_k = max(1, round(step_frac_k * (n_steps - n) + min_k))
                else:
                    step_top_k = max(1, round(step_frac_k * n + min_k))
            except:
                step_top_k = top_k

            try:
                step_top_p = top_p[0]
                min_p = min(top_p)
                max_p = max(top_p)
                step_frac_p = (max_p - min_p) / float(n_steps)
                if min_p == top_p[1]:
                    step_top_p = min(1.0, max(1E-6, step_frac_p * (n_steps - n) + min_p))
                else:
                    step_top_p = min(1.0, max(1E-6, step_frac_p * n + min_p))
            except:
                step_top_p = top_p
            # todo: unify these   
            try:
                step_temperature = temperature[0]
                min_temp = min(temperature)
                max_temp = max(temperature)
                step_frac_temp = (max_temp - min_temp) / float(n_steps)
                if min_temp == temperature[1]:
                    step_temperature = min(1.0, max(1E-8, step_frac_temp * (n_steps - n) + min_temp))
                else:
                    step_temperature = min(1.0, max(1E-8, step_frac_temp * n + min_temp))
            except:
                step_temperature = temperature

            k = int(sundae_keep_tokens_per_step[n])
            if k == 0:
                # skip zero keep scheduled steps to speed things up
                # do it this way because very long schedules need small k values
                # which necessarily causes 0 to be more frequent
                continue

            # do n_unroll_steps of resampling, randomly sampling masks during the procedure
            fwd_step = n
            if n_reps_per_mask > 1:
                # roll mask forward 
                fwd_step = int(fwd_step + n_reps_per_mask)
            p = np.maximum(alpha_min, alpha_max - fwd_step * (alpha_max-alpha_min)/(eta*int(n_steps)))
            #if intermediate_corrupt:
            if not frozen_mask:
                if n % n_reps_per_mask == 0:
                    #sampled_binaries = rs.choice(2, size = C.shape, p=[p, 1-p])
                    sampled_binaries = torch.bernoulli(1. - (0 * C + p), generator=trsg).long()
                    C2 += sampled_binaries
                if n > n_steps:
                    # set final mask to all ones
                    C2[:] = 1

            # todo: always modify mask even if intermediate_corrupt is False
            # this way the model trusts different variables all the time
            # this should be close to the *best* sampling setup
            C2[C==1] = 1
            #x_cache = x
            #if initial_corrupt:
            #    x = lcl_get_random_pitches(x.shape, P)
            #    x[C2==1] = x_cache[C2==1]

            #x = model.pred(x, C2)#, temperature=temperature)

            #x_e = torch.clone(x) # torch.tensor(x).float().to(device)
            #C2_e = torch.clone(C2) # torch.tensor(C2).float().to(device)
            # passing true will noise things
            logits_x, masks = model(x, x_idx, C2)

            # dont predict just logits anymore
            # top k top p gumbel
            if swap_at_eta:
                swap_flag = n < (eta * int(n_steps))
            else:
                swap_flag = use_typical_sampling
            if use_typical_sampling and swap_flag:
                logits_x = logits_x / float(step_temperature)
                filtered_logits_x = typical_top_k_filtering(logits_x, top_k=step_top_k, top_p=step_top_p)
            else:
                logits_x = logits_x / float(step_temperature)
                filtered_logits_x = top_k_top_p_filtering(logits_x, top_k=step_top_k, top_p=step_top_p)
            x_new = lcl_gumbel_sample(filtered_logits_x, low=1).float()

            # the even-er
            p = has_been_kept_torch[:, :] / torch.sum(has_been_kept_torch[:, :], axis=0, keepdims=True)
            r_p = 1. - p
            r_p = r_p / torch.sum(r_p, axis=0, keepdims=True)

            if k > 0:
                shp = r_p.shape
                assert len(shp) == 2
                # turn it to B, T for torch.multinomial
                r_p = r_p.permute(1, 0)
                if use_evener:
                    keep_inds_torch = torch.multinomial(r_p, num_samples=k, replacement=False, generator=trsg)
                else:
                    keep_inds_torch = torch.multinomial(0. * r_p + 1. / float(shp[0]), num_samples=k, replacement=False, generator=trsg)

                # back to T, B
                keep_inds_torch = keep_inds_torch.permute(1, 0)
                # use scatter logic 
                for _ii in range(x.shape[1]):
                    x[-latent_length:, :][keep_inds_torch[:, _ii], _ii] = x_new[keep_inds_torch[:, _ii], _ii]
                    has_been_kept_torch[keep_inds_torch[:, _ii], _ii] += 1
            else:
                pass

            x[-latent_length:][C==1] = x_cache[-latent_length:][C==1]
            if keep_mask is not None:
                x[-latent_length:][keep_C==1] = x_cache[-latent_length:][keep_C==1]

            if verbose:
                print("sample_step {}".format(n))
            if return_intermediates:
                all_x.append(x.cpu().data.numpy())
                all_C.append(C2.cpu().data.numpy())
            C2 = torch.clone(C)
        if return_intermediates:
            return x, all_x, all_C
        return x
# print all valid tracks
#for _i in range(len(valid_tracks_attribution)):
    #if 0 == valid_tracks_attribution[_i][1][0]:
    #    print(_i, valid_tracks_attribution[_i][0])
index_track = 16
print(valid_tracks_attribution[index_track])
seed_track = copy.deepcopy(valid_tracks[index_track])
global_track = copy.deepcopy(seed_track).astype("int32") - min_midi_pitch

measures = 1
# for gif plots
return_intermediates = False
num_instruments_sample = num_instruments
latent_length_sample = latent_length
total_length_sample = total_length
measure_step = (latent_length_sample // num_instruments_sample)
total_measure_step = (total_length_sample // num_instruments_sample)
# cut conditioning file short
global_track = global_track[:, -total_measure_step:]
new_chorale = Chorale(global_track, model.n_classes)

new_chorale.to_image()
plt.savefig("groundtruth_separated_image.png")
plt.close()
new_chorale.to_image_combined()
plt.savefig("groundtruth_joined_image.png")
plt.close()
groundtruth_fname = "groundtruth.mid"
# Oboe, English horn, clarinet, bassoon, sounds better on timidity.
program_map_satb = [69, 70, 72, 71]
velocity_map_satb = [70, 50, 50, 65]
bpm = 50
"""
# piano
program_map_satb = [0] * 4
velocity_map_satb = [70, 50, 50, 65]
bpm = 90
"""
new_chorale.save(groundtruth_fname, program_map_satb, velocity_map_satb, bpm)

step_start = -measure_step
step_end = step_start + measure_step
if step_end == 0:
    step_end = None
elif step_end > 0:
    raise ValueError("measure_step too large for latent length! Max should be latent_length // I")

#temperature_to_test = (.6, 0.05)
temperature_to_test = (.95, 0.05)
#temperature_to_test = (1.0, 0.05)
#steps_to_test = 5 * int(latent_length_sample)
steps_to_test = 10 * int(latent_length_sample)
n_reps_per_mask_to_test = 1
n_reps_final_mask_dwell_to_test = 0
keep_to_test = .33
#top_k_to_test = (2, 5)
top_k_to_test = (20, 5)
top_p_to_test = 0.0
#seed_offset_to_test = 7945
seed_offset_to_test = 1
#seed_offset_to_test = 2
typical_sampler_to_test = True
evener_to_test = False

n_steps = steps_to_test
n_reps_per_mask = n_reps_per_mask_to_test
n_reps_final_mask_dwell = n_reps_final_mask_dwell_to_test
sundae_keep_prob = keep_to_test
top_k = top_k_to_test
top_p = top_p_to_test
use_evener = evener_to_test
use_typical_sampling = typical_sampler_to_test
temperature = temperature_to_test

global_time = time.time()
intermediate_x = None
intermediate_C = None
for _m in range(measures):
    print("Step {}".format(_m))
    step_time = time.time()
    y = copy.deepcopy(global_track[:, -total_measure_step:])
    y[:, -measure_step:] = 0
    C = 0 * y[:, -measure_step:]

    this_seed = seed_offset_to_test + _m
    ret = torch_harmonize_perceiversunmask(y, C, model,
                                           n_steps,
                                           make_batch,
                                           model.n_classes,
                                           n_reps_per_mask=n_reps_per_mask,
                                           n_reps_final_mask_dwell=n_reps_final_mask_dwell,
                                           sundae_keep_prob=sundae_keep_prob,
                                           top_k=top_k,
                                           top_p=top_p,
                                           use_evener=use_evener,
                                           use_typical_sampling=use_typical_sampling,
                                           temperature=temperature,
                                           seed=this_seed,
                                           return_intermediates=return_intermediates,
                                           verbose=True)
    if not return_intermediates:
        raw_pred = ret
    else:
        raw_pred, intermediate_x, intermediate_C = ret
    raw_pred_np = raw_pred.cpu().data.numpy()
    raw_pred_np = raw_pred_np.transpose(1, 0)
    # now B, T
    shp = raw_pred_np.shape
    final_pred = raw_pred_np.reshape(shp[0], shp[1] // num_instruments_sample, num_instruments_sample).transpose(0, 2, 1).astype("int32")
    # push from 1 min to 0
    prop_track = final_pred[0] - 1
    # take last part and add to global track

    global_track[:, step_start:step_end] = prop_track[:, step_start:step_end]
    if _m != (measures - 1):
        global_track = np.concatenate([global_track, 0 * global_track[:, step_start:step_end]], axis=1)
    end_step_time = time.time()
    print("Overall track length", global_track.shape)
    print("Step time", time.time() - step_time)
print("Global time", time.time() - global_time)
track = copy.deepcopy(global_track)
new_chorale = Chorale(track, model.n_classes)
new_chorale.to_image()
plt.savefig("sampled_separated_image.png")
plt.close()
new_chorale.to_image_combined()
plt.savefig("sampled_joined_image.png")
plt.close()
sampled_fname = "sampled.mid"
# Oboe, English horn, clarinet, bassoon, sounds better on timidity.
program_map_satb = [69, 70, 72, 71]
velocity_map_satb = [70, 50, 50, 65]
bpm = 50
"""
# piano
program_map_satb = [0] * 4
velocity_map_satb = [70, 50, 50, 65]
bpm = 50
"""

new_chorale.save(sampled_fname, program_map_satb, velocity_map_satb, bpm)
gt_cmd = "bash timidifyit.sh {}".format(groundtruth_fname)
print("Running " + gt_cmd)
os.system(gt_cmd)
s_cmd = "bash timidifyit.sh {}".format(sampled_fname)
print("Running " + s_cmd)
os.system(s_cmd)
# true global track needs midi pitch correction + plot and midify
