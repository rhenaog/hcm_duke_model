import csv
import re
import os, os.path
import numpy as np
import sys
sys.path.append("..")
import cv2
from cv2 import imread, resize
import torch
import torchvision
from utils import save_checkpoint, load_checkpoint, save_metrics, load_metrics
from torch.optim.lr_scheduler import StepLR
import echonet
import numpy as np

import torchvision
from torch.utils.data import Dataset
import torchvideo.transforms as VT

device = torch.device("cuda")

class load_video_data(Dataset):
	# not patient-level but video-level
	def __init__(self, 
				 train_val_test_flag = None,
				 chamber_view = None,
				 mean=0., std=1.,
				 pad=None,
				 noise=None,
				 select_channel=None,
				 length=None,
				 period = 2,
				 number_of_clips=None,
				 min_len = 0,
				 max_len = None,
				 frame_perm = False,
				 spatial_dim = (112, 112),
				 give_me_all_clips = False,
				 percentile_mask_by_var = None,
				 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
				 transform=torchvision.transforms.Compose([
								VT.PILVideoToTensor(ordering='TCHW'),
							])):

		self.transform = transform
		self.mean = mean
		self.std = std
		self.pad = pad
		self.noise = noise
		self.select_channel = select_channel
		self.percentile_mask_by_var = percentile_mask_by_var
		self.spatial_dim=spatial_dim
		self.length = length
		self.min_len = min_len
		self.max_len = max_len
		self.period = period
		self.number_of_clips = number_of_clips
		self.frame_perm = frame_perm
		self.give_me_all_clips = give_me_all_clips
 	
		data_list = []
		labels = []
		prob_list = []

		cases_path = '/storage/hj152-projects/DIHI_Echo/hj152_code/hcm_data/cases_' + train_val_test_flag + '_list.csv'
		with open(cases_path, 'r') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				data_list.append(row[0])
				prob_list.append(float(row[5]))
				labels.append(1)

		controls_path = '/storage/hj152-projects/DIHI_Echo/hj152_code/hcm_data/controls_' + train_val_test_flag + '_list.csv'
		with open(controls_path, 'r') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				data_list.append(row[0])
				prob_list.append(float(row[5]))
				labels.append(0)

		self.data_lists = data_list
		self.labels = labels
	
	def __len__(self):
		return len(self.data_lists)

	def __getitem__(self, index):
		'''
		return: 
			- video: (F, C, H, W) tensor
			- f: number of frames
			- target: (1,)
		'''
		video_path = self.data_lists[index]
		video = np.load(video_path)
		video = np.stack([resize(e_data, (self.spatial_dim[0],  self.spatial_dim[1]), interpolation = cv2.INTER_AREA) for e_data in video])
		video = video.astype(np.float32) # f, h, w, c
		
		one_c_video = video[:, :, :, 0]
		var_video = np.var(one_c_video, axis=0)
		masked_video = video
		masked_video[:, var_video == 0] = 0 # percentile to be 5% mask background()
		video = masked_video

		video = np.transpose(video, (3, 0, 1, 2)) # c, f, h, w
		# Add simulated noise (black out random pixels)
		# 0 represents black at this point (video has not been normalized yet)

		if self.noise is not None:
			n = video.shape[1] * video.shape[2] * video.shape[3]
			ind = np.random.choice(n, round(self.noise * n), replace=False)
			f = ind % video.shape[1]
			ind //= video.shape[1]
			i = ind % video.shape[2]
			ind //= video.shape[2]
			j = ind
			video[:, f, i, j] = 0

		# Apply normalization
		if isinstance(self.mean, (float, int)):
			video -= self.mean
		else:
			video -= self.mean.reshape(3, 1, 1, 1)

		if isinstance(self.std, (float, int)):
			video /= self.std
		else:
			video /= self.std.reshape(3, 1, 1, 1)

		c, f, h, w = video.shape
		video = np.transpose(video, (1, 2, 3, 0)) # f, h, w, c
		if self.select_channel is not None:
			video = video[:, :, :, self.select_channel]
			c = 1

		# video = np.transpose(video, ()) # c, f, h, w

		if self.transform is not None:
			video = self.transform(video) # f, c, h, w 

		# transform the target
		target = self.labels[index]
		### augmentation by length
		video = video.to(device)
		if self.give_me_all_clips:
			if f < self.length:
				video = torch.cat([video, torch.zeros((self.length - f, c, h, w)).to(device)], dim=0)
				f = self.length

			video = video.permute(1, 0, 2, 3)
			return video, f, target

		if self.period is not None:
			l = f // self.period
			
			if self.length is not None:
				if f < self.length * self.period:
					# Pad video with frames filled with zeros if too short
					# 0 represents the mean color (dark grey), since this is after normalization
					video = torch.cat([video, torch.zeros((self.length * self.period - f, c, h, w)).to(device)], dim=0)
					f, c, h, w = video.shape  # pylint: disable=E0633
				if self.number_of_clips is not None:
					start = np.random.choice(f - (self.length - 1) * self.period, self.number_of_clips)
					video = tuple(video[s + self.period * np.arange(self.length), :, :, :] for s in start)
					video = torch.cat(video, dim=0)
					f = self.length
				else:
					start = np.random.choice(f - (self.length - 1) * self.period, 1)
					video =video[start + self.period * np.arange(self.length), :, :, :]
					f = self.length
			else:
				start = 0
				video =video[start + self.period * np.arange(l), :, :, :]
				f = l
		###
		if f < self.min_len:
			video = torch.cat([video, torch.zeros((self.min_len - f, c, h, w)).to(device)], dim=0)
			f = self.min_len
		if self.max_len is not None:
			if f > self.max_len:
				video = video[:self.max_len, :, :, :]
				f = self.max_len

		#reshape to bs, n_clips, c, f, h, w
		if self.number_of_clips is not None:
		# before that, video is f, c , h, w
			video = video.view(self.number_of_clips, self.length, video.shape[1], video.shape[2], video.shape[3])
			video = video.permute(0, 2, 1, 3, 4) # n_clip, c, f, h, w
		else:
			video = video.permute(1, 0, 2, 3) # c, f, h, w

		return video, f, target
