import os
import cv2
import torch
import matplotlib.pyplot as plt
import shutil

from math import cos, pi


def save_checkpoint(state, is_best, history_path):
    filename = os.path.join(history_path, 'checkpoint.pth')                           
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(history_path, 'model_best.pth')
        shutil.copyfile(filename, best_filename)
        
        
def adjust_learning_rate_(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def Makedir(PATH, new_folder):
    PATH = os.path.join(PATH, new_folder)
    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
        print('Error Creating director')
    return PATH
    
def train_graph(epoch, history_dict, save_path):
    file_path = os.path.join(save_path, 'history_graph.png')
    
    epochs_range = range(1,epoch+1)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_dict['train']['acc'], label='Training Accuracy')
    plt.plot(epochs_range, history_dict['val']['acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_dict['train']['loss'], label='Training Loss')
    plt.plot(epochs_range, history_dict['val']['loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(file_path, dpi=100)
    print('')
    print('The train graph is saved...')
    print('')
