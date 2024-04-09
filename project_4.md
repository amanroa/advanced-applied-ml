---
layout: page
title: "Project-4"
permalink: /project_4/
---

# Project 4 - Particle Swarm and AlexNet

In this project, I have to use particle swarm to tune hyperparameters for CIFAR-10 images. I will then use those hyperparameters to increase the accuracy when run on an AlexNet model. 

First, I will show you how I loaded the CIFAR-10 data and split it into training and testing sets.

It's important to know that I am using a CPU to run this code, so it takes me longer to run the model. I determined that I was going to be using CPU with this code:

```c
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using", device, "device")
```

## Loading and Splitting CIFAR-10 Data

```c
def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    valid_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomCrop(120, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])

    # This is where I load the dataset. 
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        normalize,
    ])

    # This is where I load the dataset.
    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

train_loader, valid_loader = get_train_valid_loader(data_dir = './data',
                       batch_size = 64, augment = True,random_seed = 123)

test_loader = get_test_loader(data_dir = './data',
                              batch_size = 72)
```

I used the same test and train loader methods that we wrote in class. 

Hyperparameter tuning is important because it can improve model performance. By modifying parameters to be at their optimal levels, we can increase the performance of a model by a lot. I wanted to see just how much hyperparameter tuning could increase my accuracy, so I wanted to create and run an AlexNet model without hyperparameter tuning. 

## AlexNet Model

Here is my AlexNet model. AlexNet is a CNN used for image classification.

```c
class AlexNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
```

Before running my model, I had to define various features such as the number of classes, epochs, batch size, and learning rate. I also had to create an optimizer. When I was running my model with this batch size of 72, my computer did take a while to compute the 5 epochs that I defined.

```c
num_classes = 10
num_epochs = 5
batch_size = 72
learning_rate = 0.001

model = AlexNetModel(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
total_step = len(train_loader)
```

And to run the model, I used this code from our class notebook: 

```c
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
```
*Without* using PSO, these are the accuracy results that I got.

<img width="536" alt="Screenshot 2024-04-09 at 12 08 10 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/3d60df82-9aba-49ad-822e-c6308dde1daa">

This code also took very long to run - over an hour. I hypothesized that once I do PSO to optimize the hyperparameters, it will not only run faster, but have a higher accuracy. 

## Particle Swarm Optimization

To start off my PSO method, I had to create an optimization function. I thought that the optimization function should return the negative validation score while training the model, so I added that in as the thing to return. I also wanted to test the effects of different batch sizes, so I left that undefined. I chose to test the learning rate as well. 

```c
def obj_function(hyperparameters):
    learning_rate, batch_size = hyperparameters
    batch_size = int(round(hyperparameters[1]))
    batch_size = max(72, min(batch_size, 256)) 

    model = AlexNetModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, valid_loader = get_train_valid_loader(data_dir = './data',
                       batch_size = batch_size, augment = True,random_seed = 123)

    test_loader = get_test_loader(data_dir = './data',
                                  batch_size = batch_size)

    for epoch in range(num_epochs):
      for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
      with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        validation_score = 100 * correct / total
        print('Accuracy of the network on the {} validation images: {} %'.format(10000, validation_score))

    return -validation_score
```

This seems to be a good optimization function, based on what I have read about PSO. 

To go along with the optimization function, I wrote a PSO function, based off of what we did in class. 

```c
def PSO(num_dimensions, num_particles, max_iter,i_min=-10,i_max=10,bounds=None,w=0.5,c1=0.25,c2=0.75):
    if bounds is None:
        particles = [({'position': [np.random.uniform(i_min, i_max) for _ in range(num_dimensions)],
                    'velocity': [np.random.uniform(-1, 1) for _ in range(num_dimensions)],
                    'pbest': float('inf'),
                    'pbest_position': None})
                    for _ in range(num_particles)]
    else:
        particles = [({'position': [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_dimensions)],
                    'velocity': [np.random.uniform(-1, 1) for _ in range(num_dimensions)],
                    'pbest': float('inf'),
                    'pbest_position': None})
                    for _ in range(num_particles)]

    # Initialize global best
    gbest_value = float('inf')
    gbest_position = None

    for _ in range(max_iter):
        for particle in particles:
            position = particle['position']
            velocity = particle['velocity']

            # Calculate the current value
            current_value = obj_function(position)

            # Update personal best
            if current_value < particle['pbest']:
                particle['pbest'] = current_value
                particle['pbest_position'] = position.copy()

            # Update global best
            if current_value < gbest_value:
                gbest_value = current_value
                gbest_position = position.copy()

            # Update particle's velocity and position
            for i in range(num_dimensions):
                r1, r2 = np.random.uniform(), np.random.uniform()
                velocity[i] = w * velocity[i] + c1*r1 * (particle['pbest_position'][i] - position[i]) + c2*r2 * (gbest_position[i] - position[i])
                position[i] += velocity[i]
                # legalize the values to the provided bounds
                if bounds is not None:
                    position[i] = np.clip(position[i],bounds[i][0],bounds[i][1])

    return gbest_position, gbest_value

```

And I defined the bounds of the learning rate and batch size. I chose batch size as my second hyperparameter because I hypothesized that larger batch sizes would make the model train faster. I am not too sure on what max_iter does, however, I made it a low number in order to reduce the amount of time spent running my code.

```c
# Parameters
num_dimensions = 2
num_particles = 3
max_iter = 2
bounds = [(0.0001, 0.1), (72, 256)]

# Run PSO
best_position, best_value = PSO(num_dimensions, num_particles, max_iter,bounds=bounds)
print("Best Position:", best_position)
print("Best Value:", best_value)
```

As of the submission time of this homework, my code has been running for 2 hours. However, I believe that my implementation is not fully correct, because I have been getting extremely low accuracies so far. I have attached a screenshot of my results (the first iteration), to show what I have so far. I will try to leave this running overnight and update it before class, however, I believe that Google Colab might time out. 

<img width="541" alt="Screenshot 2024-04-09 at 3 04 26 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/b8f47cc6-e7cd-4486-8407-e2dd5d96ca6e">

## Conclusion

Although my program hasn't gotten me the desired results, I am hopeful that my code will work it's magic overnight and find the right combination of hyperparameters to get an accuracy > 84%. 

