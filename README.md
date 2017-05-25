# mathnotes
A simple classifier which uses a neural network written from scratch to identify handwritten mathematical symbols.  This was created for my term project for 15-112 at Carnegie Mellon University.

MathNotes is a program which can take in handwritten mathematical expressions and convert them to text.
It can then take that text and send it to Wolfram Alpha, and display a picture of the output. 
MathNotes can be run by opening "neuralNet.py" and running the 'run' function with no arguments.

MathNotes is dependant on, besides built-in python libraries, numpy, a linear algebra library, and 
Python Imaging Library (PIL), which supports image editing. Both of these can be installed either via
Anaconda, a scientific computing package for python, or via pip install. 
