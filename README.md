# MusiGAN
Music Generation using LSTM, GRU &amp; GAN


## Requirements

* Python 3.x
* TensorFlow 1.15
* Keras 
* Music21

-----------
    
## Introduction

Music is an art form that I am deeply fascinated with, it is a universal language that allows human beings to connect to one another. We make music to express ourselves. So, getting a computer to try and generate some music of its own might seem disruptive to such an art form at first glance, however, I believe that innovation comes when we're confronted with difficult situations and forced to confront how we approach music. 

## Project Description 
In this project I aim to feed music into a neural network, train it, and get it to generate new music afterwards. I aim to explore a plethora of different approaches and to compare the output of each one. Looking into the current leading implementations for guidance, I manage to focus on the following three:
	1. LSTM
	2. GRU
	3. GAN
I am quite aware that getting meaningful results is a long shot and the underlying purpose behind this project is to familiarize myself with the idea of generating music using computers.

I should also note that a basic understanding of music theory would come in handy for anyone who wishes to explore the topic further. 

## Project Breakdown 

1. Set up a stable working environment that can handle the intense computation needed to train the model. 
	a. Utilize AWS or Google Cloud Platform and familiarize myself with it. 
2. Build Neural Networks 
	a. Read the audio files into a format that can be understood by a computer.
	b. Pre-process and parse the data to feed into the neural network. 
	c. Build several neural models and compare different combinations of hyperparameters. 
3. Generate new music
	a. Generate a new random note sequence from the neural network.
	b. Try and utilize Google Magenta's MIDI Interface in order to deploy the model and generate novel notes in real time using a midi keyboard and Ableton. 

I believe i was, for the most part, successful in achieving these goals, however significant time was spent on getting everything to work, which is not necessarily a bad thing as the idea behind this is to gain an exhaustive overview of the practice. 

## Tools 

### Music21:

