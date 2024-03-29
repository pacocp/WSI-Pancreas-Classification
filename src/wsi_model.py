import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from resnet import *

#torch.backends.cudnn.benchmark = True

class AggregationModel(nn.Module):
    def __init__(self, resnet, resnet_dim=2048, num_outputs=2):
        super(AggregationModel, self).__init__()
        self.resnet = resnet
        self.resnet_dim = resnet_dim
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(resnet_dim, num_outputs)
        )

    def forward_extract(self, x):
        (batch_size, bag_size, c, h, w) = x.shape
        x = x.reshape(-1, c, h, w)
        features = self.resnet.forward_extract(x)
        features = features.view(batch_size, bag_size, self.resnet_dim)

        features = features.mean(dim=1)
        return features
    
    def forward(self, x):
        features = self.forward_extract(x)
        return self.fc(features)

def train(model, criterion, optimizer, dataloaders, transforms,
          save_dir='checkpoints/models/', device='cpu',
          log_interval=100, summary_writer=None, num_epochs=100, 
          scheduler=None, verbose=True,
          patience=40, return_ids=True):
    """ 
    Train classification/regression model.
        Parameters:
            model (torch.nn.Module): Pytorch model already declared.
            criterion (torch.nn): Loss function
            optimizer (torch.optim): Optimizer
            dataloaders (dict): dict containing training and validation DataLoaders
            transforms (dict): dict containing training and validation transforms
            save_dir (str): directory to save checkpoints and models.
            device (str): device to move models and data to.
            log_interval (int): 
            summary_writer (TensorboardX): to register values into tensorboard
            num_epochs (int): number of epochs of the training
            verbose (bool): whether or not to display metrics during training
            use_attention (bool): 
        Returns:
            train_results (dict): dictionary containing the labels, predictions,
                                 probabilities and accuracy of the model on the dataset.
    """
    best_acc = 0.0
    best_epoch = 0
    best_loss = np.inf
    best_outputs = {'train': [], 'val': {}}
    acc_array = {'train': [], 'val': []}
    loss_array = {'train': [], 'val': []}
    
    global_summary_step = {'train': 0, 'val': 0}

    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()
    actual_patience = 0
    accum_iter = 8
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        sizes = {'train': 0, 'val': 0}
        inputs_seen = {'train': 0, 'val': 0}
        running_outputs = {'train': [], 'val': []}
        running_labels = {'train': [], 'val': []}

        for phase in ['train', 'val']:
            if phase == 'train':
                    model.train()
            else:
                    model.eval()

            running_loss = 0.0
            
            running_corrects = 0.0
            summary_step = global_summary_step[phase]
            # for logging tensorboard
            last_running_loss = 0.0
            last_running_corrects = 0.0
            for batch_idx, batch in enumerate(tqdm(dataloaders[phase])):
                wsi = batch[0]
                labels = batch[1]
                if return_ids:
                    patient_ids = batch[2]
                size = wsi.size(0)
                labels = labels.flatten()
                labels = labels.to(device)
                wsi = wsi.to(device)
                #wsi = transforms[phase](wsi)
                with torch.set_grad_enabled(phase=='train'):
                    # Casts operations to mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = model(wsi)
                    # saving running outputs
                    running_outputs[phase].append(outputs.detach().cpu().numpy())
                    running_labels[phase].append(labels.cpu().numpy())
                    
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs, labels)
                    loss = loss / accum_iter
                    if phase == 'train':
                        scaler.scale(loss).backward()
                    if phase == 'train' and (((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloaders[phase]))):
                        # Scales the loss, and calls backward()
                        # to create scaled gradients
                        #loss.backward()
                        # Unscales gradients and calls
                        # or skips optimizer.step()
                        scaler.step(optimizer)
                        #optimizer.step()
                        # Updates the scale for next iteration
                        scaler.update()
                        optimizer.zero_grad()

                summary_step += 1
                running_loss += loss.item() * wsi.size(0)
                
                running_corrects += torch.sum(preds == labels)
                sizes[phase] += size
                inputs_seen[phase] += size

                # Emptying memory
                outputs = outputs.detach()
                loss = loss.detach()
                torch.cuda.empty_cache()

                if (summary_step % log_interval == 0):
                    loss_to_log = (running_loss - last_running_loss) / log_interval
                    acc_to_log = (running_corrects - last_running_corrects) / log_interval
                    #loss_to_log = running_loss / log_interval
                    #acc_to_log = running_corrects / log_interval
                    if summary_writer is not None:
                        summary_writer.add_scalar("{}/loss".format(phase), loss_to_log, summary_step)
                        summary_writer.add_scalar("{}/acc".format(phase), acc_to_log, summary_step)

                    last_running_loss = running_loss
                    last_running_corrects = running_corrects
                    inputs_seen[phase] = 0.0

            global_summary_step[phase] = summary_step
            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects / sizes[phase]

            loss_array[phase].append(epoch_loss)
            acc_array[phase].append(epoch_acc)
            #print(f'{phase}: real {Counter(np.stack(running_labels[phase]).flatten())}; preds {Counter(np.stack(running_outputs[phase], axis=1)[0].argmax(axis=1))}')
            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict_best.pt'))
                best_epoch = epoch
                best_outputs['val'] = running_outputs['val']
                best_outputs['train'] = running_outputs['train']
                actual_patience = 0
            else:
                actual_patience += 1
                if actual_patience > patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
            
            if summary_writer is not None:
                summary_writer.add_scalar("{}/epoch_loss".format(phase), epoch_loss, epoch)
                summary_writer.add_scalar("{}/epoch_acc".format(phase), epoch_acc, epoch)
        if actual_patience > patience:
                break

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_last.pt'))
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_dict_best.pt')))

    results = {
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_outputs_val': best_outputs['val'],
            'best_outputs_train': best_outputs['train'],
            'labels_val': running_labels['val'],
            'labels_train': running_labels['train'],
            'best_acc': best_acc
        }

    return model, results

def evaluate(model, dataloader, dataset_size, transforms, criterion,
             device='cpu', verbose=True, return_ids=True):
    """ 
    Evaluate classification model on test set
        Parameters:
            model (torch.nn.Module): Pytorch model already declared.
            dataloasder (torch.utils.data.DataLoader): dataloader with the dataset
            dataset_size (int): Size of the dataset.
            transforms (torch.nn.Sequential): Transforms to be applied to the data
            device (str): Device to move the data to. Default: cpu.
            verbose (bool): whether or not to display metrics at the end
            use_attention (bool): whether to use the attention of not
        Returns:
            test_results (dict): dictionary containing the labels, predictions,
                                 probabilities and accuracy of the model on the dataset.
    """
    model.eval()

    corrects = 0
    predictions = []
    probabilities = []
    real = []
    losses = []
    p_idxs = []
    for batch in tqdm(dataloader):        
        wsi = batch[0]
        labels = batch[1]
        if return_ids:
            patient_ids = batch[2]
        labels = labels.flatten()
        labels = labels.to(device)

        wsi = wsi.to(device)
        #wsi = transforms(wsi)
        with torch.set_grad_enabled(False):
           
            outputs = model(wsi)

            _, preds = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)
           
        predictions.append(preds.detach().to('cpu').numpy())
        corrects += torch.sum(preds == labels)
        probabilities.append(outputs.detach().to('cpu').numpy())
        real.append(labels.detach().to('cpu').numpy())
        losses.append(loss.detach().item())
        p_idxs.append(patient_ids)

    
    accuracy = corrects / dataset_size
    predictions = np.concatenate([predictions], axis=0, dtype=object)
    probabilities = np.concatenate([probabilities], axis=0, dtype=object)
    real = np.concatenate([real], axis=0, dtype=object)
    p_idxs = np.concatenate([p_idxs], axis=0, dtype=object)
    print('Accuracy of the model {}'.format(accuracy))
    print('Loss of the model {}'.format(np.mean(losses)))
    
    test_results = {
        'outputs': probabilities,
        'real': real,
        'accuracy': accuracy.detach().to('cpu').numpy(),
        'predictions': predictions,
        'patient_ids': p_idxs
    }

    return test_results
