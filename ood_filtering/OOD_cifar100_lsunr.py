import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np
from collections import Counter
import math
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

# import warnings
# warnings.filterwarnings("ignore")

x_axis = [7000]
geometric_list = []
D_star_geometric_list = []
data_removed = []

ind_dataset = 'cifar100'
ood_dataset = 'lsunr'
num_iid = 7000
model_name = 'WideResNet'

for x in x_axis:
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(42) 

    # Set up parameters
    n_samples_OOD = x

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Load CIFAR-10 as the InD dataset
    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Split CIFAR-10 into two halves: 25,000 for training and 25,000 for the wild dataset
    cifar100_indices = list(range(len(cifar100_train)))
    random.shuffle(cifar100_indices)
    train_indices = cifar100_indices[:25000]
    wild_indices = cifar100_indices[50000-num_iid:]

    train_data_inD = Subset(cifar100_train, train_indices)
    wild_data_inD = Subset(cifar100_train, wild_indices)

    # Extract corresponding CIFAR-10 labels
    cifar100_labels = torch.tensor([cifar100_train.targets[idx] for idx in wild_indices])

    # # Load lsun_r as the OOD dataset
    transform_ood = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std),
                                                           transforms.RandomCrop(32, padding=4)])
    svhn_train = torchvision.datasets.ImageFolder(root='./data/LSUN', transform=transform_ood)

    print("\n\nlsun_r_train:", len(svhn_train))

    idx = np.array(range(len(svhn_train)))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    train_len = int(0.7 * len(svhn_train))
    aux_subset_idxs = idx[:train_len]
    test_subset_idxs = idx[train_len:]

    test_svhn_train = torch.utils.data.Subset(svhn_train, test_subset_idxs)
    data_OOD = torch.utils.data.Subset(svhn_train, aux_subset_idxs)
    ood_labels = torch.full((len(data_OOD),), -1)
    
    print("\n\nWild data (InD) dimensions:")
    sample, label = wild_data_inD[0]
    print(f"Data shape: {sample.shape}")
    print(f"Number of samples: {len(wild_data_inD)}")
    print(f"Label shape: {label}")  

    print("\nTrain data (InD) dimensions:")
    sample, label = data_OOD[0]
    print(f"Data shape: {sample.shape}")
    print(f"Number of samples: {len(train_data_inD)}")
    print(f"Label shape: {label}")  

    # Combine CIFAR-10 wild data with OOD data to form the wild mixture data
    wild_data = ConcatDataset([wild_data_inD, data_OOD])
    wild_labels = torch.cat([cifar100_labels, ood_labels])

    unique_labels_before = Counter(wild_labels.tolist())
    print(f"\n\n\nwild_data: {len(wild_data)}")
    print(f"wild_labels: {wild_labels.shape[0]}")
    print("Unique labels before:", *[f"{i}:{unique_labels_before[i]}" for i in range(100)], f"-1:{unique_labels_before[-1]}" if -1 in unique_labels_before else "")
    print("train_data_inD:", train_data_inD.dataset.data[train_data_inD.indices].shape)
    print("train_labels_inD:", len(train_data_inD.indices))


    # # start code for training the InD classifier
    hyperparams = {
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'n_epochs': 100,
    'batch_size': 128,
    'dropout_rate': 0.3,
    'num_classes' : 100,
    'checkpoint_dir': 'saved_checkpoint'}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class BasicBlock(nn.Module):
        def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
            super(BasicBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
            self.droprate = dropRate
            self.equalInOut = (in_planes == out_planes)
            self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                    padding=0, bias=False) or None

        def forward(self, x):
            if not self.equalInOut:
                x = self.relu1(self.bn1(x))
            else:
                out = self.relu1(self.bn1(x))
            if self.equalInOut:
                out = self.relu2(self.bn2(self.conv1(out)))
            else:
                out = self.relu2(self.bn2(self.conv1(x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
            if not self.equalInOut:
                return torch.add(self.convShortcut(x), out)
            else:
                return torch.add(x, out)

    class NetworkBlock(nn.Module):
        def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
            super(NetworkBlock, self).__init__()
            self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

        def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
            layers = []
            for i in range(nb_layers):
                layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
            return nn.Sequential(*layers)

        def forward(self, x):
            return self.layer(x)

    # Wide ResNet for CIFAR-100 classification
    class WideResNet(nn.Module):
        def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
            super(WideResNet, self).__init__()
            nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
            assert ((depth - 4) % 6 == 0)
            n = (depth - 4) // 6
            block = BasicBlock
            # 1st conv before any network block
            self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                                padding=1, bias=False)
            # 1st block
            self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
            # 2nd block
            self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
            # 3rd block
            self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
            # global average pooling and classifier
            self.bn1 = nn.BatchNorm2d(nChannels[3])
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(nChannels[3], num_classes)
            self.nChannels = nChannels[3]

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

        def forward(self, x):
            out = self.conv1(x)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.nChannels)
            return self.fc(out)


    model = WideResNet(40, hyperparams['num_classes'], 2, dropRate=hyperparams['dropout_rate']).cuda()

    optimizer = optim.SGD(
        model.parameters(), 
        lr=hyperparams['learning_rate'], 
        momentum=hyperparams['momentum'], 
        weight_decay=hyperparams['weight_decay']
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=hyperparams['n_epochs'])
    criterion = nn.CrossEntropyLoss()
    train_loader_inD = DataLoader(train_data_inD, batch_size=hyperparams['batch_size'], shuffle=True)

    # Create directory to save checkpoints if not exists
    os.makedirs(hyperparams['checkpoint_dir'], exist_ok=True)
    # checkpoint_path = os.path.join(hyperparams['checkpoint_dir'], 'cifar100_wide_resnet_checkpoint_nov6.pth')
    checkpoint_path =  'saved_checkpoint/cifar100_wrn_pretrained_epoch_99.pt'
    print('\ncheckpoint_path: ', checkpoint_path)

    # # Training loop
    # for epoch in range(hyperparams['n_epochs']):
    #     model.train()
    #     running_loss = 0.0
    #     for i, (inputs, labels) in enumerate(train_loader_inD):
    #         inputs, labels = inputs.cuda(), labels.cuda()

    #         optimizer.zero_grad()

    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
    #     scheduler.step()
    #     print(f"Epoch [{epoch+1}/{hyperparams['n_epochs']}], Loss: {running_loss/len(train_loader_inD):.4f}")
        
    # # Save the final model checkpoint
    # torch.save(model.state_dict(), checkpoint_path)

    # # Load the model and test the model on the training data
    # model.load_state_dict(torch.load(checkpoint_path))
    # model.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for inputs, labels in train_loader_inD:
    #         inputs, labels = inputs.cuda(), labels.cuda()
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # accuracy = 100 * correct / total
    # print(f'Accuracy on training data: {accuracy:.4f}')
    # # end code for training the InD classifier


    ## start code to get predicted labels for wild data
    # wild_loader will use the ConcatDataset
    wild_loader = DataLoader(wild_data, batch_size=hyperparams['batch_size'], shuffle=False)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    predicted_labels = []
    with torch.no_grad():  
        for inputs, _ in wild_loader:  #
            inputs = inputs.cuda() 
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability

            predicted_labels.append(preds.cpu())
    predicted_labels = torch.cat(predicted_labels)
    ## end code to get predicted labels for wild data



    # Start code for computing avg. of InD gradients
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    def get_penultimate_layer(model):
        # Check if the model is a WideResNet instance
        if isinstance(model, WideResNet):
            return model.bn1  # Return the penultimate layer before the fully connected layer
        else:
            raise ValueError("Model structure not recognized. Please inspect the printed structure and modify this function accordingly.")

    # Get the penultimate layer
    try:
        penultimate_layer = get_penultimate_layer(model)
        print(f"\nPenultimate layer: {penultimate_layer}")
    except ValueError as e:
        print(e)
        print("Please modify the get_penultimate_layer function based on the printed model structure.")
        exit(1)

    # computing gradients only for the penultimate layer
    for param in model.parameters():
        param.requires_grad = False
    for param in penultimate_layer.parameters():
        param.requires_grad = True

    def compute_avg_gradients(model, dataloader, penultimate_layer):
        gradients = []
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Collect gradients from the penultimate layer
            layer_gradients = []
            for param in penultimate_layer.parameters():
                if param.grad is not None:
                    layer_gradients.append(param.grad.view(-1))
            
            if layer_gradients:
                # gradients.append(torch.cat(layer_gradients))
                gradients.append(torch.tensor(layer_gradients[-1]))

        # Compute average gradient
        if gradients:
            avg_gradient = torch.stack(gradients).mean(dim=0)
            return avg_gradient
        else:
            return None

    avg_gradient_inD = compute_avg_gradients(model, train_loader_inD, penultimate_layer)
    print("\nAverage InD gradients shape:", avg_gradient_inD.shape)
    # End code for computing avg. of InD gradients

    # # Start code for computing Wild gradients
    wild_loader = DataLoader(wild_data, batch_size=1, shuffle=False) 
    gradients_list = []
    count = 0
    for inputs, _ in wild_loader:
        count += 1
        inputs = inputs.cuda()
        inputs.requires_grad = True
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability
        loss = criterion(outputs, preds)
        model.zero_grad()
        loss.backward()
        
        # Collect gradients from the penultimate layer
        layer_gradients = []
        for param in penultimate_layer.parameters():
            if param.grad is not None:
                layer_gradients.append(param.grad.view(-1))
       
        # Stack and append to the list if there are gradients
        if layer_gradients:
            # gradients_list.append(torch.cat(layer_gradients).cpu()) 
            gradients_list.append(torch.tensor(layer_gradients[-1]).cpu())  # Flatten and move to CPU

    concat_gradients = torch.stack(gradients_list).to(avg_gradient_inD.device)
    # # End code for computing Wild gradients


    ## start code to compute l2 distance
    def weiszfeld_algorithm(points, max_iterations=1000, tolerance=1e-7):
        # Use mean instead of median for initial guess
        # median = torch.median(points, dim=0).values
        median = torch.mean(points, dim=0)
        for _ in range(max_iterations):
            distances = torch.norm(points - median, dim=1)
            weights = 1 / torch.clamp(distances, min=1e-8)
            new_median = torch.sum(weights.unsqueeze(1) * points, dim=0) / torch.sum(weights)
            if torch.norm(new_median - median) < tolerance:
                break
            median = new_median
        return median

    concat_gradients_cpu = concat_gradients.cpu()
    avg_gradient_inD_cpu = avg_gradient_inD.cpu()

    # geometric_median = weiszfeld_algorithm(concat_gradients_cpu)
    epsilon = 0.005
    max_iterations = 100  # Set a maximum number of iterations to prevent infinite loops
    top_k = 7000 

    # if ind=7k, ood=7k, epsilon = 0.0024658, iteration-1 top_k = 7000, then data_removed: [0.7214], 5050/7000 samples removed

    prev_l2_distance_all = 1.0e-8
    iteration = 0

    wild_data = torch.cat([data.unsqueeze(0) for data, _ in wild_data], dim=0)
    outliers_data = torch.empty(0, *wild_data.shape[1:], dtype=wild_data.dtype)
    outliers_labels = torch.empty(0, dtype=wild_labels.dtype)

    while True:
        # geometric_median = weiszfeld_algorithm(concat_gradients_cpu)
        geometric_median = torch.median(concat_gradients_cpu, dim=0).values
        l2_distance_all = torch.norm(geometric_median - avg_gradient_inD_cpu)
        
        geometric_list.append(torch.norm(geometric_median).item())
        D_star_geometric_list.append(l2_distance_all.item())

        print(f"\nIteration {iteration + 1}")
        print(f"geometric_median size: {geometric_median.size()}")
        print(f"L2 distance D*: {l2_distance_all.item()}")
        print(f"GM norm: {torch.norm(geometric_median).item()}")
        print(f"avg_gradient_inD_cpu[0:3]: {avg_gradient_inD_cpu[0:3]}")

        l2_distances = []
        for i in range(concat_gradients_cpu.shape[0]):
            mask = torch.ones(concat_gradients_cpu.shape[0], dtype=torch.bool)
            mask[i] = False
            
            reduced_gradients = concat_gradients_cpu[mask]
            geometric_median = torch.median(reduced_gradients, dim=0).values

            l2_distance = torch.norm(geometric_median - avg_gradient_inD_cpu)
            difference = l2_distance_all - l2_distance 
            l2_distances.append(difference.item())

            if (i + 1) % 2000 == 0:
                print(f"Processed {i + 1} samples")

        l2_distances_tensor = torch.tensor(l2_distances)
        top_k_indices = torch.argsort(l2_distances_tensor, descending=True)[:top_k]
        top_k_indices_sorted = sorted(top_k_indices.tolist(), reverse=True)

        # Extract outliers based on sorted top-k indices
        new_outliers_data = torch.cat([wild_data[idx].unsqueeze(0) for idx in top_k_indices_sorted], dim=0)
        new_outliers_labels = torch.cat([wild_labels[idx].unsqueeze(0) for idx in top_k_indices_sorted], dim=0)

        # Concatenate the new outliers with previously accumulated outliers
        outliers_data = torch.cat((outliers_data, new_outliers_data), dim=0)
        outliers_labels = torch.cat((outliers_labels, new_outliers_labels), dim=0)

        # Create a mask to retain only non-outlier samples
        mask = torch.ones(wild_data.size(0), dtype=torch.bool)
        mask[top_k_indices_sorted] = False

        # Apply the mask to keep only inliers in the original tensors
        concat_gradients_cpu = concat_gradients_cpu[mask]
        wild_data = wild_data[mask]
        wild_labels = wild_labels[mask]

        print(f"New shape of concat_gradients_cpu: {concat_gradients_cpu.shape}")
        print(f"New shape of wild_data: {wild_data.shape}")
        print(f"New shape of wild_labels: {wild_labels.shape}")
        print(f"abs(prev_l2_distance_all - l2_distance_all): {abs(prev_l2_distance_all - l2_distance_all)}")
        unique_labels_after = Counter(wild_labels.tolist())
        data_removed.append(1 - unique_labels_after.get(-1, 0) / unique_labels_before.get(-1, 1))
        print("data_removed:", data_removed)
        print("Unique labels after:", *[f"{i}:{unique_labels_after[i]}" for i in range(10)], f"-1:{unique_labels_after[-1]}" if -1 in unique_labels_after else "")
        
        unique_labels_added = Counter(outliers_labels.tolist())
        print("Unique labels added:", *[f"{i}:{unique_labels_added[i]}" for i in range(10)], f"-1:{unique_labels_added[-1]}" if -1 in unique_labels_added else "")
        
        # Directory path to save the wild data
        save_dir = 'saved_data'

        # Create folder if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save wild data and labels tensors
        torch.save(outliers_data, os.path.join(save_dir, 'wild_data_InD{}-{}_OOD{}-{}_model-{}_topk{}_ep{}.pt'.format(num_iid, ind_dataset, n_samples_OOD, ood_dataset, model_name, top_k, epsilon)))
        torch.save(outliers_labels, os.path.join(save_dir, 'wild_labels_InD{}-{}_OOD{}-{}_model-{}_topk{}_ep{}.pt'.format(num_iid, ind_dataset, n_samples_OOD, ood_dataset, model_name, top_k, epsilon)))
        print("Wild data saved successfully in 'saved_data' folder.")

        # Check convergence
        if abs(prev_l2_distance_all - l2_distance_all) < epsilon:
            print(f"\nConverged after {iteration + 1} iterations.")
            break

        prev_l2_distance_all = l2_distance_all
        iteration += 1

        
        if iteration >= max_iterations:
            print(f"\nReached maximum number of iterations ({max_iterations}).")
            break
        ## end code to compute l2 distance of wild data

print(f"GM_norm_list: {geometric_list}")
print(f"D_star_geometric_list: {D_star_geometric_list}")
print(f"data_removed: {data_removed}")
print(f"x_axis: {x_axis}")

print(f"\nwild_labels: {wild_labels.size()}")
print(f"wild_data: {wild_data.size()}")

print(f"\noutliers_data: {outliers_data.size()}")
print(f"outliers_labels: {outliers_labels.size()}")


