import argparse
import os
import torch
import numpy as np
import pickle as pkl
import subprocess
import json
import random
from tqdm import tqdm
from scipy import linalg
from pathlib import Path
import pandas as pd

import scipy.stats as stats

# source: https://github.com/m-hamza-mughal/convofusion/blob/main/quant_eval/metric_eval.py
class FIDCalculator(object):
    @staticmethod
    def frechet_distance(samples_A, samples_B):
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)
        try:
            frechet_dist = FIDCalculator.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
        except ValueError:
            frechet_dist = 1e+10
        return frechet_dist


    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                    inception net (like returned by the function 'get_predictions')
                    for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                    representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                    representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                    'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
        
'''def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov
'''
def calculate_avg_distance(feature_list, mean=None, std=None):
    #feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_avg_distance(feature_list, mean=None, std=None):
    #feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist


class SRGR(object):
    def __init__(self, threshold=0.1, joints=63):
        self.threshold = threshold
        self.pose_dimes = joints
        self.counter = 0
        self.sum = 0

    def run(self, results, targets, semantic):
        results = results.reshape(-1, self.pose_dimes, 3)
        targets = targets.reshape(-1, self.pose_dimes, 3)
        semantic = semantic.reshape(-1)
        diff = np.sum(abs(results - targets), 2)
        success = np.where(diff < self.threshold, 1.0, 0.0)
        for i in range(success.shape[0]):
            # srgr == 0.165 when all success, scale range to [0, 1]
            success[i, :] *= semantic[i] * (1 / 0.165)
        rate = np.sum(success) / (success.shape[0] * success.shape[1])
        self.counter += success.shape[0]
        self.sum += rate * success.shape[0]
        return rate

    def avg(self):
        return self.sum / self.counter


class L1div(object):
    def __init__(self):
        self.counter = 0
        self.sum = 0

    def run(self, results):
        self.counter += results.shape[0]
        mean = np.mean(results, 0)
        for i in range(results.shape[0]):
            results[i, :] = abs(results[i, :] - mean)
        sum_l1 = np.sum(results)
        self.sum += sum_l1

    def avg(self):
        return self.sum / self.counter

import librosa
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
from data.dnd_config import *
class alignment(object):
    def __init__(self, sigma, order):
        self.sigma = sigma
        self.order = order
        self.times = self.oenv = self.S = self.rms = None
        self.pose_data = []
    
    def load_audio(self, wave, t_start, t_end, without_file=False, sr_audio=16000):
        if without_file:
            y = wave
            sr = sr_audio
        else: y, sr = librosa.load(wave, sr=sr_audio)
        short_y = y #[int(t_start*sr):int(t_end*sr)]
        self.oenv = librosa.onset.onset_strength(y=short_y, sr=sr)
        self.times = librosa.times_like(self.oenv)
        # Detect events without backtracking
        onset_raw = librosa.onset.onset_detect(onset_envelope=self.oenv, backtrack=False)
        if len(onset_raw) == 0:
            # print(len(wave))
            return None, None, None
        onset_bt = librosa.onset.onset_backtrack(onset_raw, self.oenv)
        self.S = np.abs(librosa.stft(y=short_y))
        self.rms = librosa.feature.rms(S=self.S)
        onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, self.rms[0])
        return onset_raw, onset_bt, onset_bt_rms
    
    def load_pose(self, pose, t_start, t_end, pose_fps, without_file=False):
        # pose: T, J, 3
        data_each_file = pose #.reshape(-1, 189//3, 3)
        vel= data_each_file[1:] - data_each_file[:-1]
        
        right_shoulder = vel[:, DND_JOINT_NAMES.index('RightShoulder')]
        vel_right_shoulder = np.linalg.norm(right_shoulder, axis=-1)
        
        right_arm = vel[:, DND_JOINT_NAMES.index('RightArm')]
        vel_right_arm = np.linalg.norm(right_arm, axis=-1)
        
        right_wrist = vel[:, DND_JOINT_NAMES.index('RightHand')]
        vel_right_wrist = np.linalg.norm(right_wrist, axis=-1)
        
        beat_right_arm = argrelextrema(vel_right_arm, np.less, order=self.order)
        beat_right_shoulder = argrelextrema(vel_right_shoulder, np.less, order=self.order)
        beat_right_wrist = argrelextrema(vel_right_wrist, np.less, order=self.order)
        
        left_shoulder = vel[:, DND_JOINT_NAMES.index('LeftShoulder')]
        vel_left_shoulder = np.linalg.norm(left_shoulder, axis=-1)
        
        left_arm = vel[:, DND_JOINT_NAMES.index('LeftArm')]
        vel_left_arm = np.linalg.norm(left_arm, axis=-1)
        
        left_wrist = vel[:, DND_JOINT_NAMES.index('LeftHand')]
        vel_left_wrist = np.linalg.norm(left_wrist, axis=-1)
        
        beat_left_arm = argrelextrema(vel_left_arm, np.less, order=self.order)
        beat_left_shoulder = argrelextrema(vel_left_shoulder, np.less, order=self.order)
        beat_left_wrist = argrelextrema(vel_left_wrist, np.less, order=self.order)
        
        return beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist
    
    def load_data(self, wave, pose, t_start, t_end, pose_fps):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, t_start, t_end, pose_fps)
        return onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist 

    def eval_random_pose(self, wave, pose, t_start, t_end, pose_fps, num_random=60):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        dur = t_end - t_start
        for i in range(num_random):
            beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, i, i+dur, pose_fps)
            dis_all_b2a= self.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist)
            print(f"{i}s: ",dis_all_b2a)

    def audio_beat_vis(self, onset_raw, onset_bt, onset_bt_rms):
        figure(figsize=(24, 6), dpi=80)
        fig, ax = plt.subplots(nrows=4, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(self.S, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[0])
        ax[0].label_outer()
        ax[1].plot(self.times, self.oenv, label='Onset strength')
        ax[1].vlines(librosa.frames_to_time(onset_raw), 0, self.oenv.max(), label='Raw onsets', color='r')
        ax[1].legend()
        ax[1].label_outer()

        ax[2].plot(self.times, self.oenv, label='Onset strength')
        ax[2].vlines(librosa.frames_to_time(onset_bt), 0, self.oenv.max(), label='Backtracked', color='r')
        ax[2].legend()
        ax[2].label_outer()

        ax[3].plot(self.times, self.rms[0], label='RMS')
        ax[3].vlines(librosa.frames_to_time(onset_bt_rms), 0, self.oenv.max(), label='Backtracked (RMS)', color='r')
        ax[3].legend()
        fig.savefig("./onset.png", dpi=500)
    
    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        time_vel = vel[0]/pose_fps + offset 
        return time_vel    
    
    @staticmethod
    def GAHR(a, b, sigma):
        dis_all_a2b = 0
        dis_all_b2a = 0
        for b_each in b:
            l2_min = np.inf
            for a_each in a:
                l2_dis = abs(a_each - b_each)
                if l2_dis < l2_min:
                    l2_min = l2_dis
            dis_all_b2a += math.exp(-(l2_min**2)/(2*sigma**2))
        dis_all_b2a /= len(b)
        return dis_all_b2a 

    def calculate_align(self, onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, sr, pose_fps=25):

        audio_bt = librosa.frames_to_time(onset_bt_rms, sr = sr)
        pose_bt = self.motion_frames2time(beat_right_wrist, 0, pose_fps)
        avg_dis_all_b2a = self.GAHR(pose_bt, audio_bt, self.sigma)
        return avg_dis_all_b2a  
    
    def calculate_align_(self, onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, sr, pose_fps=25):
        # more stable solution
        avg_dis_all_b2a = 0
        for audio_beat in [onset_bt_rms]: #[onset_raw, onset_bt, onset_bt_rms]:
            for pose_beat in [beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist]:
                audio_bt = librosa.frames_to_time(audio_beat, sr = sr)
                pose_bt = self.motion_frames2time(pose_beat, 0, pose_fps)
                dis_all_b2a = self.GAHR(pose_bt, audio_bt, self.sigma)
                avg_dis_all_b2a += dis_all_b2a
        avg_dis_all_b2a /= 6 #18
        return avg_dis_all_b2a 


def angular_difference(v1, v2, eps=1e-6):
    # B, T, 3
    # Normalize the vectors
    v1_norm = torch.nn.functional.normalize(v1, dim=-1, eps=eps)
    v2_norm = torch.nn.functional.normalize(v2, dim=-1, eps=eps)

    # Compute dot product
    dot = (v1_norm * v2_norm).sum(dim=-1).clamp(-1.0, 1.0)  # Clamp for numerical stability

    # Compute angle in radians, then convert to degrees
    angles_rad = torch.acos(dot)
    angles_deg = torch.rad2deg(angles_rad)

    return angles_deg if angles_deg.numel() > 1 else angles_deg.item()