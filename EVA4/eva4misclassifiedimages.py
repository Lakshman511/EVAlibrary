import matplotlib.pyplot as plt
import torch
import numpy as np
device="cuda"
channel_means = (0.49139968, 0.48215841, 0.44653091)
channel_stdevs = (0.24703223, 0.24348513, 0.26158784)
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
def unnormalize(img):
  img = img.cpu().numpy().astype(dtype=np.float32)
  
  for i in range(img.shape[0]):
    img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
  return np.transpose(img, (1,2,0))
  
# a class to maintain misclassified images,train loss,test loss values and also to plot graphs
class StatsManager:
  def __init__(self):
    self.data={"mis_classified_images":[],"pred":[],"target":[],"train_loss":[],"test_loss":[],"train_accuracy":[],"test_accuracy":[]}
  def append_train_loss(self,loss):
    self.data["train_loss"].append(loss)
  def append_test_loss(self,loss):
    self.data["test_loss"].append(loss)
  def append_train_accuracy(self,acc):
    self.data["train_accuracy"].append(acc)
  def append_test_accuracy(self,acc):
    self.data["test_accuracy"].append(acc)	
  def append_misclassified_images(self,data1,pred,target):
    for j in range(len(pred)):
      if(pred[j]!=target[j]):
        self.data['pred'].append(pred[j])
        self.data['target'].append(target[j])
        self.data["mis_classified_images"].append(data1[j,:,:,:])
  def give_misclassified_25(self):
  	l=[]
  	for j in self.data.mis_classified_images[-1:-26:-1]:
  		l.append(j.numpy())
  def plot_misclassified_25(self):
    figure = plt.figure(figsize=(10,10))
    num_of_images = 25
    for index in range(1, num_of_images + 1):
      plt.subplot(5, 5, index)
      plt.axis('off')
      plt.annotate("Pred: "+class_names[self.data["pred"][-index].cpu().item()]+" Tar:"+class_names[self.data["target"][-index].cpu().item()],(3,3),bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
      plt.imshow(unnormalize(self.data["mis_classified_images"][-index]), interpolation='none')
  def plot_losses(self):
    plt.figure(figsize=(15,10))
    plt.plot(self.data["train_loss"])
    plt.plot(self.data["test_loss"])
    plt.title('model losses')
    plt.ylabel('losses')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
  def plot_accuracies(self):
    plt.figure(figsize=(15,10))
    plt.plot(self.data["train_accuracy"])
    plt.plot(self.data["test_accuracy"])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()