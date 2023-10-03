import sys
import pygame
import ctypes
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize as re
from utils import *

# Pygame config
pygame.init()
fps = 300
fpsClock = pygame.time.Clock()
width, height = 1280, 960
screen = pygame.display.set_mode((width, height))
font = pygame.font.SysFont('Arial', 20)
resFont = pygame.font.SysFont('Arial', 25)
# Increase Dots Per inch so it looks sharper
ctypes.windll.shcore.SetProcessDpiAwareness(True)

# Variables
# The model we are visualizing
model = Network([])
# Our Buttons will append themselves to this list
objects = []
# Initial color
drawColor  = [255, 255, 255]
shadeColor = [150, 150, 150]
# Initial brush size
brushSize = 18
brushSizeSteps = 3
# Drawing Area Size
canvasSize = [488, 488]
# Button Variables.
buttonWidth  = 120
buttonHeight = 35
# Canvas
canvas = pygame.Surface(canvasSize)
canvas.fill((0, 0, 0))
# Store the model guesses
guesses = np.zeros(10)

# Button Class
class Button():
    def __init__(self, x, y, width, height, buttonText='Button', onclickFunction=None, onePress=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.onclickFunction = onclickFunction
        self.onePress = onePress
        self.fillColors = {
            'normal' : '#ffffff',
            'hover'  : '#666666',
            'pressed': '#333333',
        }
        self.buttonSurface = pygame.Surface((self.width, self.height))
        self.buttonRect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.buttonSurf = font.render(buttonText, True, (20, 20, 20))
        self.alreadyPressed = False
        objects.append(self)

    def process(self):
        mousePos = pygame.mouse.get_pos()
        self.buttonSurface.fill(self.fillColors['normal'])
        if self.buttonRect.collidepoint(mousePos):
            self.buttonSurface.fill(self.fillColors['hover'])
            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                self.buttonSurface.fill(self.fillColors['pressed'])
                if self.onePress:
                    self.onclickFunction()
                elif not self.alreadyPressed:
                    self.onclickFunction()
                    self.alreadyPressed = True
            else:
                self.alreadyPressed = False
        self.buttonSurface.blit(self.buttonSurf, [
            self.buttonRect.width/2 - self.buttonSurf.get_rect().width/2,
            self.buttonRect.height/2 - self.buttonSurf.get_rect().height/2
        ])
        screen.blit(self.buttonSurface, self.buttonRect)

# Handler Functions
# Changing the Color
def changeColor(color):
    global drawColor
    drawColor = color

# Changing the Brush Size
def changebrushSize(dir):
    global brushSize
    if dir == 'greater':
        brushSize += brushSizeSteps
    else:
        brushSize -= brushSizeSteps

# Save the surface to the Disk
def save():
    pygame.image.save(canvas, "canvas.png")

# Clear the canvas
def clear():
    canvas.fill((0, 0, 0))

# Convert the cavas into a test case
def extract():
    image = np.zeros((canvasSize[0],canvasSize[1]))
    for col in range(canvasSize[0]):
        for row in range(canvasSize[1]):
            image[row,col] = canvas.get_at((col,row))[0]
    prcimg = lib_downsample(image)
    return prcimg

# Pass the drawing through the network
# returns the array of guesses that the network has
def run(network):
    img = extract()
    output = network.forward_pass(img)
    global guesses
    guesses = output
    return output

# Pass the drawing through the network
# returns the array of guesses that the network has
def train(network, label):
    img = extract()
    network.train(img, label)
    run(network)

# resize an image using a library method
def lib_downsample(image):
    resized_image = re(image, (28, 28), anti_aliasing=True)
    return resized_image

def show_image(image):
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show()

def show_nn(image, CNN):
    current_image = image[:,:,np.newaxis]/255.
    CvL1_output   = CNN.layers[0].forward_prop(current_image)
    MPL1_output   = CNN.layers[1].forward_prop(CvL1_output)
    ReLu_output   = CNN.layers[2].forward_prop(MPL1_output)
    CvL2_output   = CNN.layers[3].forward_prop(ReLu_output)
    MPL2_output   = CNN.layers[4].forward_prop(CvL2_output)
    prediction    = CNN.layers[5].forward_prop(MPL2_output)
    ncols         = 6
    fig, ax       = plt.subplots(8, ncols)
    # Image
    ax[0][ncols-1].imshow(current_image, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    # Convolutions
    for i in range(min(CvL1_output.shape[2], 8)):
        ax[i][0].imshow(CvL1_output[:,:,i], cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    # Max Pooling
    for i in range(min(MPL1_output.shape[2], 8)):
        ax[i][1].imshow(MPL1_output[:,:,i], cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    # ReLu
    for i in range(min(ReLu_output.shape[2], 8)):
        ax[i][2].imshow(ReLu_output[:,:,i], cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    # Convolutions 2
    for i in range(min(CvL2_output.shape[2], 8)):
        ax[i][3].imshow(CvL2_output[:,:,i], cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    # Max Pooling 2
    for i in range(min(MPL2_output.shape[2], 8)):
        ax[i][4].imshow(MPL2_output[:,:,i], cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    # Softmax
    ax[1][ncols-1].imshow(prediction[:,np.newaxis], cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.show()

def build_buttons(network):
    global model
    model = network
    # Buttons and their respective functions.
    buttons = [
        ['Black', lambda: changeColor([0, 0, 0])],
        ['White', lambda: changeColor([255, 255, 255])],
        ['Clear', clear],
        ['Brush Larger', lambda: changebrushSize('greater')],
        ['Brush Smaller', lambda: changebrushSize('smaller')],
        # ['Save', save],
        ['Run', lambda: run(network)],
    ]
    # Making the buttons
    for index, buttonName in enumerate(buttons):
        Button(index * (buttonWidth + 10) + 10, 10, buttonWidth,
            buttonHeight, buttonName[0], buttonName[1])
    build_training_buttons(network)

def build_training_buttons(network):
    buttons = [
        ['0', lambda: train(network, 0)],
        ['1', lambda: train(network, 1)],
        ['2', lambda: train(network, 2)],
        ['3', lambda: train(network, 3)],
        ['4', lambda: train(network, 4)],
        ['5', lambda: train(network, 5)],
        ['6', lambda: train(network, 6)],
        ['7', lambda: train(network, 7)],
        ['8', lambda: train(network, 8)],
        ['9', lambda: train(network, 9)],
    ]
    # Making the buttons
    for index, buttonName in enumerate(buttons):
        Button((width/2) + 20, index * 80 + 95, buttonWidth/2,
            buttonHeight, buttonName[0], buttonName[1])

def loop():
    draw_cd = 0
    # main loop
    while True:
        screen.fill((30, 30, 30))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # Drawing the Buttons
        for object in objects:
            object.process()
        # Draw the Canvas at the center of the screen
        x, y = screen.get_size()
        screen.blit(canvas, [x/4 - canvasSize[0]/2, y/2 - canvasSize[1]/2])
        # Find top guess:
        most_likely = 0.0
        most_likely_index = 0
        for i, val in enumerate(guesses):
            if val > most_likely:
                most_likely = val
                most_likely_index = i
        # Printing model output
        for i, val in enumerate(guesses):
            if i != most_likely_index:
                screen.blit(resFont.render('{}: {:.1f}%'.format(i, val*100), False, 'white'), [(width/2)+100+buttonWidth, 80+80*i])
            else:
                screen.blit(resFont.render('{}: {:.1f}%'.format(i, val*100), False, 'green'), [(width/2)+100+buttonWidth, 80+80*i])
        # Drawing with the mouse
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            # Calculate Position on the Canvas
            dx = mx - x/4 + canvasSize[0]/2
            dy = my - y/2 + canvasSize[1]/2
            pygame.draw.circle(
                canvas,
                drawColor,
                [dx, dy],
                brushSize
            )
            draw_cd = 60
        # Reference Dot
        pygame.draw.circle(
            screen,
            drawColor,
            [100, 100],
            brushSize,
        )
        pygame.display.flip()
        fpsClock.tick(fps)
        draw_cd = max(0, draw_cd-1)
        if draw_cd <= 0:
            run(model)
            draw_cd = 600
