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
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

class MyRobot():
    def __init__(self, robot):
        
        self.robot = robot
    
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.right_motor.setPosition(float('inf'))
        self.right_motor.setVelocity(0.0)
        
        self.timestep = int(self.robot.getBasicTimeStep())
        
        self.camera = self.robot.getDevice('camera')
        Camera.enable(self.camera, self.timestep)

class Expert(MyRobot):
    def __init__(self, robot):
        super().__init__(robot)
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
        self.actions = []
        self.states = []
    
    def keyboard_teleop(self, key, left_speed, right_speed, speed):
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
        
    def run(self):
        left_speed = 0.0
        right_speed = 0.0
        speed = 0.0
        quit = False
        path = "data/"
        print("Expert on control...")
        while self.robot.step(self.timestep) != -1:
            image = self.camera.getImageArray()
            motor_speed, speed, quit = self.keyboard_teleop(self.keyboard.getKey(),left_speed, right_speed, speed)
            self.left_motor.setVelocity(motor_speed[0])
            self.right_motor.setVelocity(motor_speed[1])
            self.actions.append(motor_speed.copy())
            self.states.append(image.copy())
            if quit:
                print("Saving data...")
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                np.save(path + "actions.npy", self.actions)
                np.save(path + "states.npy", self.states)
                print("Data saved!")
                break

class Net(nn.Module):

    def __init__(self, epochs, lr, batch_size):
        super(Net, self).__init__()
        
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        
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
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr) # Optimization algorithm
        
        # Compute model on selected device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Current device: {self.device}")
        self.to(self.device) 
        self.path = "data/"
        self.trainset = None
        self.testset = None
        self.mae = []
        self.cost = []
        

    def forward(self, x):
        x = x.to(self.device) # Send data to selected device
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = torch.flatten(x, 1) # Linear needs 1 demensional vector
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        return x
    
    def train_model(self):
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
        epoch_mae = 0
        epoch_loss = 0

        print("Training model...")
        for i in range(self.epochs):
            for x, y in self.trainset:

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
    
            for x, y in self.testset:
                # Send data to selected device
                x = x.to(torch.device(self.device))
                y = y.to(torch.device(self.device))
                y_pred = self(x)
                
                target = torch.sigmoid(y).cpu().detach()
                pred = y_pred.cpu().detach()
                
                epoch_mae += self.compute_mae(pred, target)
                
            epoch_mae /= len(self.testset.dataset) # Mean absolute error
            epoch_loss /= len(self.trainset.dataset) # Total loss

            self.mae.append(epoch_mae) # Records the accuracy for each epoch
            self.cost.append(epoch_loss) # Records the loss for each epoch
            
            print(f"Epoch {i}: loss: {epoch_loss:.4f} - MAE: {epoch_mae:.4f}")

        torch.save(self.state_dict(), self.path+"trained_model.pth")
        print("Model trained!")
        self.plot_results()
        
    def load_data(self):
        """
        Load data from data/actions.npy and data/states.npy.
        Transform data to tensor and create a DataLoader.
        Split data between train data and test data.
        """
        print("Loading data...")
        # load data from .npy and trasform to tensor.
        actions = torch.from_numpy(np.load(self.path+"actions.npy")).to(torch.float32)
        states = torch.from_numpy(np.load(self.path+"states.npy")).to(torch.float32)
        dataset = TensorDataset(states.transpose(1,3),actions)
        
        # split data between train data and test data.
        split = 0.75
        train_batch = int(split * len(dataset))
        test_batch = len(dataset) - train_batch
        train_dataset, test_dataset = random_split(dataset, [train_batch, test_batch])
    
        # create a DataLoader.
        self.trainset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.testset = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def compute_mae(self, pred, target):
        result = mean_absolute_error(target.numpy(), pred.numpy())
        return result
        
    def plot_results(self):
        plt.plot(range(self.epochs), self.cost)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function Value x Epochs')
        plt.grid()
        plt.savefig("plot_cost_function_over_epochs.png")
        plt.show()
        plt.clf()
        plt.plot(range(self.epochs), self.mae)
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error x Epochs')
        plt.grid()
        plt.savefig("plot_mae_over_epochs.png")
        plt.show()  
        
class Agent(MyRobot):
    def __init__(self, net, robot):
        super().__init__(robot)
        self.net = net
       
    def run(self, need_train):
        if need_train:
            self.net.load_data()
            self.net.train() # training mode
            self.net.train_model()
        self.net.load_state_dict(torch.load("data/trained_model.pth")) # load trained model
        self.net.eval() # evaluation mode
        with torch.no_grad(): # disable gradient calculation for inference
            while self.robot.step(self.timestep) != -1:
                image = self.camera.getImageArray()
                data = torch.from_numpy(np.array(image).copy()).to(torch.float32) 
                data = data.unsqueeze(0)
                data = data.transpose(1,3)
                motor_speed = torch.logit(self.net(data)).squeeze().tolist()
                if motor_speed[0]>20:
                    motor_speed[0]=20
                elif motor_speed[0]<-20:
                    motor_speed[0]=-20
                if motor_speed[1]>20:
                    motor_speed[1]=20
                elif motor_speed[1]<-20:
                    motor_speed[1]=-20
                self.left_motor.setVelocity(motor_speed[0])
                self.right_motor.setVelocity(motor_speed[1])
                # print(motor_speed)     
    

if __name__ == "__main__":
    robot = Robot()    
    expert = Expert(robot)
    net = Net(100, 0.0001, 50)
    agent = Agent(net, robot)
        
    need_data = False
    need_train = True
    
    if need_data:
        expert.run()
    agent.run(need_train)