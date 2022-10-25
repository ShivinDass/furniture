#!/usr/bin/env python
# coding: utf-8

# Load the gamepad and time libraries
import Gamepad
import time

# Gamepad settings
gamepadType = Gamepad.PS4
circle_button = 'CIRCLE'
r1_button = 'R1'
r2_button = 'R2'
l1_button = 'L1'
l2_button = 'L2'
joyleft_x = 'LEFT-X'
joyleft_y = 'LEFT-Y'
joyright_x = 'RIGHT-X'
joyright_y = 'RIGHT-Y'

# Wait for a connection
if not Gamepad.available():
    print('Please connect your gamepad...')
    while not Gamepad.available():
        time.sleep(1.0)
gamepad = gamepadType()
print('Gamepad connected')

# Handle joystick updates one at a time
while gamepad.isConnected():
    # Wait for the next event
    eventType, control, value = gamepad.getNextEvent()

    # Determine the type
    if eventType == 'BUTTON':
        # Button changed
        if control == circle_button:
            if value:
                print('CIRCLE')
        elif control == r1_button:
            if value:
                print('R1')
        elif control == l1_button:
            if value:
                print('L1')
    elif eventType == 'AXIS':
        # Joystick changed
        if control == joyright_x:
            print('RightX: ', value)
        elif control == joyright_y:
            print('RightY: ', value)
