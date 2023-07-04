"""expert_controller controller."""

#!/usr/bin/env python3

from controller import Robot
from controller import Camera
from controller import Keyboard
import numpy as np

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
    

keyboard = Keyboard()
keyboard.enable(timestep)

left_speed = 0.0
right_speed = 0.0
speed = 0.0
quit = False
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

    
    
    