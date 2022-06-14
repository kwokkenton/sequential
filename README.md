# Sequential

My attempt at learning processing sequential data: recurrent neural networks (RNNs), time series data and natural language processing.

I am following the Dive Into Deep Learning syllabus as the primary reference. However, I have chosen not to import the entire d2l library and copy out individual functions from the source code instead, as I believe that it would help my understanding. These helper functions are in `utils.py`. 

The chapters referenced are
8. Recurrent Neural Networks
9. Modern Recurrent Neural Networks
10. Attention Mechanisms
14. Natural Language Processing: Pretraining
15. Natural Language Processing: Applications

In addition to running the code, my attempts on the exercises are included. 

## Contents
`sequence_model.ipynb`: investigation of using a Multi-Layer Perceptron model to forecast a time series. Looks at an autoregressive model to extrapolate a sinusoid using various k-step ahead models. Main takeaway is that errors accumulate when this happens. Refers to Chapter 8.1 of d2l.ai. 
`text_models.ipynb`: Refers to Chapters 8.2-3 of d2l.ai. 
`rnn.ipynb`: Refers to the implementation of a recurrent neural network from scratch.  
`concise_rnn.ipynb`: Uses high level api in Pytorch to code the same RNN. Refers to Chapters 8.6-7 of d2l.ai. 

## Additional relevant materials:
* Stanford [Lecture 8: Recurrent Neural Networks and Language Models](https://www.youtube.com/watch?v=Keqep_PKrY8&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=8) on the issue of vanishing gradients and the chain rule 
* https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks