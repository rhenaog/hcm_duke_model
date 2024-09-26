import csv
import re
import tqdm
import os, os.path
import numpy as np
from sklearn import metrics
import time
import subprocess
import sys
sys.path.append("..")
import torchvision
from utils import save_checkpoint, load_checkpoint, save_metrics, load_metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvideo.transforms as VT

from data_loader_hcm import load_video_data

mean = np.array([15.890775, 48.323906, 48.834034])
std = np.array([33.840668, 62.168327, 62.729694])

period =1
length = 16
spatial_dim = (112, 112)
frame_perm=False
jitter_ratio = 0.5
train_transform = transform=torch.transforms.Compose([
	VT.PILVideoToTensor(ordering='TCHW'),
	])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_val_test_flag = 'train'
train_dataset = load_video_data(
	train_val_test_flag = train_val_test_flag,     
	mean = mean, std = std,
	noise = 0.02,
	length=length,
	min_len = length,
	max_len = length,
	period = period,
	select_channel = None,
	frame_perm = frame_perm,
	transform = train_transform,
	spatial_dim = spatial_dim,
	device=device)

train_val_test_flag = 'val'
valid_dataset = load_video_data(
	train_val_test_flag = train_val_test_flag,      
	mean = mean, std = std,
	noise = None,
	length=length,
	min_len = length,
	max_len = length,
	period = period,
	select_channel = None,
	frame_perm = False,
	spatial_dim = spatial_dim,
	give_me_all_clips = True,
	device=device)
  
print("Train num {}, Valid num {}".format(len(train_dataset), len(valid_dataset)))

# dataloader
train_batch_size = 16
valid_batch_size = 1

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = valid_batch_size, shuffle = False)

num_targets = 2

model = torchvision.models.video.r2plus1d_18(pretrained=True)
device = torch.device("cuda")
model = torch.nn.DataParallel(model)

ct = 0
for child in model.module.children():
	ct += 1
	if ct <= 1:
		for param in child.parameters():
			param.requires_grad = False

model.module.fc = nn.Linear(in_features=model.module.fc.in_features, out_features=2, bias=True)

file_dir = "/file_path_to_hcm_groups/"
model_dir = "/save_model_dir/"
if not os.path.exists(model_dir):
	os.makedirs(model_dir)
print(model_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-3
optimizer = torch.optim.SGD(model.module.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3, eta_min = 1e-4)

criterion = nn.CrossEntropyLoss()
m = nn.Sigmoid()
num_epochs = 200
eval_every_epochs = 1
eval_every = len(train_dataloader) * eval_every_epochs
file_path = model_dir
saved_path = model_dir
best_valid_loss = float("Inf")
training_show_every = 200
clip = 7.0
train_loader = train_dataloader
valid_loader = valid_dataloader
###
# 2D prefunc CNN tr
model = model.to(device)
sample_count = 0.0
running_acc = 0.0
running_loss = 0.0
valid_running_acc = 0.0
valid_running_loss = 0.0
global_step = 0
train_acc_list = []
valid_acc_list = []
train_loss_list = []
valid_loss_list = []
global_steps_list = []

best_valid_auc = 0

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

top_cut = 20
bottom_cut = 90
right_cut = 30
left_cut = 90
target_size = (112,112)
for epoch in range(num_epochs):
	start_time = time.time()

	model.train()
	sample_count = 0.0
	running_acc = 0.0
	running_loss = 0.0
	all_logits = []
	ground_truth_list = [] 
	for videos, frame_nums, labels in tqdm.tqdm(train_dataloader):
		videos = videos.type(torch.FloatTensor).to(device)
		frame_nums = frame_nums.type(torch.LongTensor).to(device)
		labels = labels.type(torch.LongTensor).to(device)
		# view_prob = view_prob.type(torch.FloatTensor).to(device)

		cropped_img = []
		for i in range(videos.shape[0]):
			new_data = videos[i,:,:,top_cut:bottom_cut,right_cut:left_cut]
			new_data = new_data.permute(1, 0, 2, 3)
			resized_swapped_tensor = F.interpolate(new_data, size=target_size, mode='bicubic', align_corners=False)
			resized_tensor = resized_swapped_tensor.permute(1, 0, 2, 3)
			resized_tensor = torch.unsqueeze(resized_tensor,0)
			cropped_img.append(resized_tensor)

		cropped_img = torch.cat(cropped_img,dim=0)
		output = model(cropped_img)
		loss = criterion(output, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# update running values
		pred_labels = torch.argmax(output.data.cpu(), dim=1)
		logits = F.softmax(output.data, dim=1)
		ground_truth_list.append(labels.cpu())
		all_logits.append(logits.cpu().numpy())
		running_acc += torch.sum(pred_labels.t().squeeze() == labels.data.cpu().squeeze()).item()
		sample_count += labels.shape[0]
		running_loss += loss.item()
		global_step += 1

		# training stats
		if global_step % training_show_every == 0:
			average_running_acc = running_acc / sample_count
			average_train_loss =  running_loss / sample_count
			print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}'
				.format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
					average_train_loss, average_running_acc))
	average_train_acc = running_acc / sample_count
	average_train_loss =  running_loss / sample_count
	logits_list = np.concatenate(all_logits)
	label = np.concatenate(ground_truth_list).ravel()
	fpr, tpr, thresh = metrics.roc_curve(label,logits_list[:,1])
	train_auc = metrics.auc(fpr,tpr)
	
	# evaluation step
	if (epoch+1) % eval_every_epochs == 0:
		model.eval()
		with torch.no_grad():

			# validation loop view level
			print("Begin Validation")
			valid_sample_count = 0.0
			valid_running_loss = 0.0
			valid_running_acc = 0.0
			valid_labels = []
			valid_outputs = []
			all_logits = []
			ground_truth_list = []
			for videos, frame_nums, labels in tqdm.tqdm(valid_dataloader):
				videos = videos.type(torch.FloatTensor).to(device)
				frame_nums = frame_nums.type(torch.LongTensor).to(device)
				labels = labels.type(torch.LongTensor).to(device)
				# view_prob = view_prob.type(torch.FloatTensor).to(device)

				loop_num = int(np.floor(frame_nums.data.cpu().numpy()/16))
				sub_output = []
				sub_logits = []
				for i in range(loop_num):
					video_sub = videos[:,:,16*i:16*(i+1),:,:]
					cropped_img = []
					for i in range(video_sub.shape[0]):
						new_data = video_sub[i,:,:,top_cut:bottom_cut,right_cut:left_cut]
						new_data = new_data.permute(1, 0, 2, 3)
						resized_swapped_tensor = F.interpolate(new_data, size=target_size, mode='bicubic', align_corners=False)
						resized_tensor = resized_swapped_tensor.permute(1, 0, 2, 3)
						resized_tensor = torch.unsqueeze(resized_tensor,0)
					cropped_img.append(resized_tensor)					
					cropped_img = torch.cat(cropped_img,dim=0)
					output = model(cropped_img)
					logits = F.softmax(output,dim=1)
					sub_logits.append(logits)
					sub_output.append(output)

				labels = labels.reshape(-1,1)

				max_out = torch.mean(torch.cat(sub_output,dim=0),dim=0)
				max_logits = torch.mean(torch.cat(sub_logits,dim=0),dim=0)
				weighted_out = max_logits[1]

				manual_loss = -torch.mean(((1-labels.float()) * torch.log(1-weighted_out) + labels.float() * torch.log(weighted_out)), axis=0)
				loss = manual_loss[0]

				valid_labels.append(labels)
				valid_outputs.append(weighted_out)
				valid_running_loss += loss.item()

				pred_labels = weighted_out >= 0.5
				pred_labels = pred_labels.int()
					
				ground_truth_list.append(labels.cpu())
				all_logits.append(weighted_out.detach().cpu().numpy())
				valid_running_loss += (pred_labels == labels).sum().item()
				valid_sample_count += labels.shape[0]
				valid_running_acc += (pred_labels == labels).sum().item()

		valid_labels = torch.cat(valid_labels, dim=0)
		valid_outputs = torch.stack(valid_outputs, dim=0)
		valid_outputs_np = valid_outputs.cpu().data.numpy()
		valid_labels_np = valid_labels.cpu().data.numpy()				
		############

		############
		# evaluation
		label = np.concatenate(ground_truth_list).ravel()
		fpr, tpr, thresh = metrics.roc_curve(valid_labels_np,valid_outputs_np)
		val_auc = metrics.auc(fpr,tpr) 

		average_valid_loss = valid_running_loss / len(valid_dataloader)
		average_valid_acc = valid_running_acc / valid_sample_count

		train_loss_list.append(average_train_loss)
		train_acc_list.append(average_train_acc)
		valid_loss_list.append(average_valid_loss)
		valid_acc_list.append(average_valid_acc)
		global_steps_list.append(global_step)

		# resetting running values
		running_loss = 0.0
		valid_running_loss = 0.0
		# print progress
		video_data_list = []
		split_flag = '2'

		cases_path = file_dir + '/cases_val_list.csv'
		with open(cases_path, 'r') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				video_data_list.append(row[0])

		controls_path = file_dir + '/controls_val_list.csv'
		with open(controls_path, 'r') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				video_data_list.append(row[0])

		study_level_prediction = {}
		for idx, e_data in enumerate(video_data_list): # validation list
			study_name = e_data.split('/')[-1].split('_')[0]
			if valid_labels_np[idx] == 0:
				study_name = "/controls/" + study_name
			else:
				study_name = "/cases/" + study_name
			if study_name not in study_level_prediction:
				study_level_prediction[study_name] = ([], valid_labels_np[idx])
				study_level_prediction[study_name][0].append(valid_outputs_np[idx])
			else:
				study_level_prediction[study_name][0].append(valid_outputs_np[idx])
				assert study_level_prediction[study_name][1] == valid_labels_np[idx]

		study_level_prediction_list = []
		study_level_prediction_label = []
		for e_study in study_level_prediction:
			preds = study_level_prediction[e_study][0]
			study_pred = np.zeros(1)
			for e_pred in preds:
				study_pred += e_pred
			study_pred = study_pred/len(preds)
			study_level_prediction_list.append(study_pred)
			study_level_prediction_label.append(study_level_prediction[e_study][1])	

		study_level_prediction_list = np.vstack(study_level_prediction_list)
		study_level_prediction_label = np.array(study_level_prediction_label)		
		fpr, tpr, thresh = metrics.roc_curve(study_level_prediction_label,study_level_prediction_list)
		val_study_auc = metrics.auc(fpr,tpr) 

		# resetting running values
		running_loss = 0.0
		valid_running_loss = 0.0

		print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Train AUC: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f}, Val AUC: {:.4f}, Val Study AUC: {:.4f}'
			.format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
				average_train_loss, average_train_acc, train_auc, average_valid_loss, average_valid_acc, val_auc, val_study_auc))
		print("Max memory used: {} Mb ".format(torch.cuda.memory_allocated(device=0)/ (1024 * 1024)))

		# checkpoint
		if best_valid_auc <= val_auc:
			best_valid_auc = val_auc 
			save_checkpoint(os.path.join(saved_path, 'best_epoch_' + str(epoch) + '.pt'), model, val_auc)

	scheduler.step()
