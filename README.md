# MusiGAN
Music Generation using LSTM, GRU &amp; GAN


## Requirements

* Python 3.x
* TensorFlow 1.15/Keras 2.2.4-tf
* Keras 
* Music21
* Magenta

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

1. Set up a stable working environment that can handle the intense computation needed to train the model. \
	a. Utilize AWS or Google Cloud Platform and familiarize myself with it. <br />
2. Build Neural Networks <br />
	a. Read the audio files into a format that can be understood by a computer. <br />
	b. Pre-process and parse the data to feed into the neural network. <br />
	c. Build several neural models and compare different combinations of hyperparameters. <br />
3. Generate new music <br />
	a. Generate a new random note sequence from the neural network. <br />
	b. Try and utilize Google Magenta's MIDI Interface in order to deploy the model and generate novel notes in real time using a midi keyboard and Ableton. <br />

I believe i was, for the most part, successful in achieving these goals, however significant time was spent on getting everything to work, which is not necessarily a bad thing as the idea behind this is to gain an exhaustive overview of the practice. 

## Packages

_Music21_: A package written by MIT faculty, it is a set of tools for helping scholars and other active listeners answer questions about music quickly and simply. I utilize it in order to convert 

_Keras/TensorFlow_: In order to build my models I utilized CuDNNLSTM and CuDNNGRU from Keras.
	
CuDNNLSTM: Fast LSTM implementation backed by cuDNN. 
	
	"Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies."
	
CuDNNGRU: Fast GRU implementation backed by cuDNN.
	
	"The GRU, known as the Gated Recurrent Unit is an RNN architecture, which is similar to LSTM units. The GRU comprises of the reset gate and the update gate 		instead of the input, output and forget gate of the LSTM.
	The reset gate determines how to combine the new input with the previous memory, and the update gate defines how much of the previous memory to keep around. If 	we set the reset to all 1’s and update gate to all 0’s we again arrive at our plain RNN model."

## Data

Firstly, we need to use audio files in MIDI (*.mid) in order to be able to feed it into our different models. MIDI files are basically a set of instructions. MIDI data contains a list of events or messages that tell an electronic device (musical instrument, computer sound card, cell phone, et cetera) how to generate a certain sound. 

Although many exhaustive datasets exist with thousands of labelled midi files, I decided to go for a relatively basic dataset, bearing in mind that the goal is not to beat some benchmark accuracy, but to be able try out a lot of different things to see what works and what doesn't. Moreover, I came to the realization that the dataset used is quite important when it comes to the performance of a model. Eventually, I decided to go for the *Pokemon MIDIs* dataset as it not heavy, quite simple in terms of melody.

The Process was as follows: <br />
	1. Convert files melodic sequences into a sequence of notes. <br />
	2. Extract different elements from the song, i.e. pitch, dominant key, etc. <br />
	3. The notes were then converted into a list of corresponding integer indices (as Neural Networks handle numerical data better than categorical data) <br />
	4. Feed into network <br />

## Results &amp; Findings

-- Many online tutorials online exist that show you how to generate music using RNN models. I used these as a springboard in order to kickstart my project. In addition to  Having built many models sing different hyperprameters, it became clear to me that generating music GANs, generated better results. I shall be going over the model details duirng the presentation. Furthermore, I shall aim to utilize majenta.js in order to showcase my model generating music in real time. 

-- I found that myself leaning towards GANs and their output for two main reasons: a) They were much faster to train b) I was getting repetitive notes using LSTM/GRU. In terms of model evaluation, I found it interesting to listen to the different outputs and comparing them to one another, irrespective of score. I placed less emphasis on some numerical metric and more effort into trying to manipulate the output as much as I can. Irrespective of the model used, the actual output itself, I believe, would need a fair amount of human manipulation in order to turn it into "music"

-- Preprocessing the data has a substantial effect on model performance, i.e. note frequency, rhythm pattern, etc. However, this sort of data manipulation would specific be specific to the data/instrument being fed into the network; such modifications may not be appropriate for another dataset/instrument.

## Future Work 

In the future i would like to work some more on  building a more robust GAN model but with a greater emphasis on the output, i.e. generate two MIDI sequences, transpose one of them, and then play them using instruments. For some reason, I have been unsuccessful in generating music using other instruments than the piano using Music21's instrument library. I would like to see if the instrument library can be somehow configured into our model. Furthermore, suppose we have a track with two instruments playing simultaneously in one track, would it be possible to somehow train a isolate the two instruments, feed them into their respective models, but still somehow link both of them together? Perhaps assigning some evaluation metric could be utilized in order for the two models to take one another into account during training. 



	
