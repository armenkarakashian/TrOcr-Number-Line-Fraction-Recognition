from jiwer import cer
import jiwer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset
import requests
import json
import cv2
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image

def remove_vinculum(s):
	return s.replace("|", "").replace(" ", "")

def evaluate_model(model, dataset, processor):
	model.eval()
	total_cer = 0
	n_samples = len(dataset)

	transformation = jiwer.Compose([
	    jiwer.ToLowerCase(),
	    jiwer.Strip(),
	    jiwer.RemoveSpecificWords("|")  # Add more characters if needed
	])

	with torch.no_grad():
		counter = 0
		for sample in dataset:
			pixel_values = sample['pixel_values'].unsqueeze(0).to(model.device)
			labels = sample['labels'].unsqueeze(0).to(model.device)
			image_name = sample['image_name']

			#	Generate predictions
			outputs = model.generate(pixel_values, eos_token_id=2, num_return_sequences=4)
			
			#	Get actual label
			actual_labels = labels.tolist()[0]  # Convert tensor to list
			actual_labels = [label for label in actual_labels if label != -100]  # Filter out the -100s
			
			pretransformation_label = processor.batch_decode([actual_labels], skip_special_tokens=True)[0]
			decoded_labels = remove_vinculum(pretransformation_label)
			
			decoded_preds = processor.batch_decode(outputs, skip_special_tokens=True)
			lowest_cer = 1.1
			best_pred = None
			for pred in decoded_preds:
				pretransformation_pred = pred
				transformed_pred = remove_vinculum(pretransformation_pred)

				#	Compute CER 
				sample_cer = cer(decoded_labels, transformed_pred)
				if sample_cer < lowest_cer:
					best_pred = pretransformation_pred
					lowest_cer = sample_cer
				
				
			total_cer += lowest_cer

			#	Print the results for each image
			print(f"Image Name: {image_name}")
			print(f"Actual Text: {pretransformation_label}")
			print(f"Predicted Text: {best_pred}")
			print(f"CER: {lowest_cer}\n")
			counter+=1
			if counter == 118:
				break

	#	Average metrics
	avg_cer = total_cer / n_samples

	return {"CER": avg_cer}

#	Step 1:
#	Prepare data
class IAMDataset(Dataset):
	def __init__(self, root_dir, df, processor, max_target_length=128):
		self.root_dir = root_dir
		self.df = df
		self.processor = processor
		self.max_target_length = max_target_length

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		# get file name + text 
		file_name = self.df['filename'][idx]
		text = self.df['label'][idx]
		# prepare image (i.e. resize + normalize)
		image = Image.open(self.root_dir + file_name).convert("RGB")
		pixel_values = self.processor(image, return_tensors="pt").pixel_values
		# add labels (input_ids) by encoding the text
		labels = self.processor.tokenizer(text,padding="max_length",max_length=self.max_target_length).input_ids
		# important: make sure that PAD tokens are ignored by the loss function
		labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
		encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels), "image_name": file_name}
		return encoding
	
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-stage1")
test_df = pd.read_csv('test.csv', usecols=[0, 1])

test_dataset = IAMDataset(root_dir='', df=test_df, processor=processor)
model = VisionEncoderDecoderModel.from_pretrained('RF_MODEL/checkpoint-1400')

model.config.eos_token_id = 2

# Call the evaluate function
metrics = evaluate_model(model, test_dataset, processor)
print(metrics)

