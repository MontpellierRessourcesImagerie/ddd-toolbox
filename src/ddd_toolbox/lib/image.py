import numpy as np


class ImageCalculator(object):


    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2
        self.operation = 'add'
        self.result = None


    def run(self):
        if self.operation == "add":
            self.add()
        if self.operation == "subtract":
            self.subtract()
        if self.operation == "multiply":
            self.multiply()
        if self.operation == "divide":
            self.divide()


    def add(self):
        self.result = np.add(self.image1, self.image2)


    def subtract(self):
        self.result = np.subtract(self.image1, self.image2)


    def multiply(self):
        self.result = np.multiply(self.image1, self.image2)


    def divide(self):
        self.result = np.divide(self.image1, self.image2)