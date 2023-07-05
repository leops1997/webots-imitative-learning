"""expert_controller controller."""

#!/usr/bin/env python3

from controller import Robot
from controller import Camera
from controller import Keyboard

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transform
import matplotlib.pyplot as plt

robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice('left wheel motor')
left_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)

right_motor = robot.getDevice('right wheel motor')
right_motor.setPosition(float('inf'))
right_motor.setVelocity(0.0)

camera = robot.getDevice('camera')
Camera.enable(camera, timestep)

path = "./"
actions = []
states = []

def load_data():
    """
    Load data from data/actions.npy and data/states.npy.
    Transform data to tensor and create a DataLoader.
    Split data between train data and test data.
    """

    # load data from .npy and trasform to tensor.
    path = "data/"
    actions = torch.from_numpy(np.load(path+"actions.npy")).to(torch.float32)
    states = torch.from_numpy(np.load(path+"states.npy")).to(torch.float32)
    dataset = TensorDataset(states.transpose(1,3),actions)
    
    # split data between train data and test data.
    split = 0.75
    train_batch = int(split * len(dataset))
    test_batch = len(dataset) - train_batch
    train_dataset, test_dataset = random_split(dataset, [train_batch, test_batch])

    # create a DataLoader.
    batch_size = 1
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Build network        
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(8,8), stride=4)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 512)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 2)
        self.act4 = nn.Sigmoid()
        
        self.loss_function = nn.BCELoss() # Binary Cross Entropy
        self.optimizer = optim.Adam(self.parameters(), lr=0.00001) # Optimization algorithm
        
        # Compute model on selected device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Current device: {self.device}")
        self.to(self.device) 

    def forward(self, x):
        x = x.to(self.device) # Send data to selected device
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = torch.flatten(x, 1) # Linear needs 1 demensional vector
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        return x
    
    def train_model(self, trainset, testset, epochs, path):
        """
        Train model with the trainset.
        Test model with the testset.
        Train the model by the amount of epochs.
        Save model on "path/trained_model.pth"
        :param trainset: training dataset
        :param testset: test dataset
        :param epochs: number os epochs
        :param path: path to save the model
        :return cost: List with the cost of each epoch
        :return accuracy: List with the accuracy of each epoch
        """
        epoch_accuracy = 0
        epoch_loss = 0
        accuracy = []
        cost = []
        for i in range(epochs):
            for x, y in trainset:

                # Send data to selected device
                x = x.to(torch.device(self.device))
                y = y.to(torch.device(self.device))

                # Forward
                y_pred = self(x)
                target_normalized = torch.sigmoid(y) # Put target values between 0 and 1
                loss = self.loss_function(y_pred, target_normalized)
                self.optimizer.zero_grad() # Clear accumulated gradients

                epoch_loss += loss.item() # Sum loss

                # Backward
                loss.backward()
                self.optimizer.step() # Update parametes
    
            for x, y in testset:
                # Send data to selected device
                x = x.to(torch.device(self.device))
                y = y.to(torch.device(self.device))
                y_pred = self(x)

                epoch_accuracy += ((torch.sigmoid(y) - y_pred) ** 2).sum().item() # Mean squared error

            epoch_accuracy /= len(testset.dataset) # Mean squared error
            epoch_loss /= len(trainset.dataset) # Total loss

            accuracy.append(epoch_accuracy) # Records the accuracy for each epoch
            cost.append(epoch_loss) # Records the loss for each epoch
            
            print(f"Epoch {i}: loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

        torch.save(self.state_dict(), path+"trained_model.pth")

        return cost, accuracy


def keyboard_teleop(key, left_speed, right_speed, speed):
    left_speed = speed
    right_speed = speed
    quit = False
    if key == Keyboard.UP:
        speed = speed + 1.0
        if speed > 20: speed = 20
        left_speed = speed
        right_speed = speed
    if key == Keyboard.DOWN:
        speed = speed - 1.0
        if speed < -20: speed = -20
        left_speed = speed
        right_speed = speed
    if key == Keyboard.LEFT:
        left_speed = speed * 0.5
    if key == Keyboard.RIGHT:
        right_speed = speed * 0.5
    if key == Keyboard.END:
        quit = True
    return [left_speed, right_speed], speed, quit
 
    
if __name__ == "__main__":

    keyboard = Keyboard()
    keyboard.enable(timestep)
    
    left_speed = 0.0
    right_speed = 0.0
    speed = 0.0
    quit = False
    path = "data/"
    while robot.step(timestep) != -1:
        image = camera.getImageArray()
        motor_speed, speed, quit = keyboard_teleop(keyboard.getKey(),left_speed, right_speed, speed)
        left_motor.setVelocity(motor_speed[0])
        right_motor.setVelocity(motor_speed[1])
        actions.append(motor_speed.copy())
        states.append(image.copy())
        if quit:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            np.save(path + "actions.npy",actions)
            np.save(path + "states.npy", states)
            break
    train_dataset, test_dataset = load_data() 
    net = Net()
    net.train() # training mode
    net.train_model(train_dataset, test_dataset, 20, "data/")
    net.load_state_dict(torch.load("data/trained_model.pth")) # load trained model
    net.eval() # evaluation mode
    with torch.no_grad(): # disable gradient calculation for inference
        while robot.step(timestep) != -1:
            image = camera.getImageArray()
            data = torch.from_numpy(np.array(image).copy()).to(torch.float32) 
            data = data.unsqueeze(0)
            data = data.transpose(1,3)
            motor_speed = torch.logit(net(data)).squeeze().tolist()
            left_motor.setVelocity(motor_speed[0])
            right_motor.setVelocity(motor_speed[1])
            print(motor_speed)               
    