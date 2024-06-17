import re
import json

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

train_loss_pattern = r'Train Loss: (\d+\.\d+)'
train_acc_pattern = r'Train Acc: (\d+\.\d+)'
test_loss_pattern = r'Test Loss: (\d+\.\d+)'
test_acc_pattern = r'Test Acc: (\d+\.\d+)'

file_folder = 'origin_optimizer/new'
file_name = '--optim_type Adamax'
file_path = './' + file_folder + '/' + file_name + '.txt'
with open(file_path, 'r') as file:
    for line in file:
        train_loss_match = re.search(train_loss_pattern, line)
        if train_loss_match:
            train_loss_list.append(float(train_loss_match.group(1)))
        train_acc_match = re.search(train_acc_pattern, line)
        if train_acc_match:
            train_acc_list.append(float(train_acc_match.group(1)))
        test_loss_match = re.search(test_loss_pattern, line)
        if test_loss_match:
            test_loss_list.append(float(test_loss_match.group(1)))
        test_acc_match = re.search(test_acc_pattern, line)
        if test_acc_match:
            test_acc_list.append(float(test_acc_match.group(1)))

dict = {"train_loss": train_loss_list, "train_acc": train_acc_list, "test_loss": test_loss_list, "test_acc": test_acc_list, "file_folder": file_folder, "file_name": file_name}

json_path = './' + file_folder + '/' + file_name + '.json'
with open(json_path, 'w') as f:
    json.dump(dict, f)