# Sequential

My attempt at learning processing sequential data: recurrent neural networks (RNNs), time series data and natural language processing.

I am following the Dive Into Deep Learning syllabus (Chapters 8-10) as the primary reference. However, I have chosen not to import the entire d2l library and copy out individual functions from the source code instead, as I believe that it would help my understanding. These helper functions are in `utils.py`. 

In addition to running the code, my attempts on the exercises are included. 

Additional relevant materials:
* Stanford [Lecture 8: Recurrent Neural Networks and Language Models](https://www.youtube.com/watch?v=Keqep_PKrY8&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=8) on the issue of vanishing gradients and the chain rule 
* https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
  
## Contents
`sequence_model.ipynb`: investigation of using a Multi-Layer Perceptron model to forecast a time series. Looks at an autoregressive model to extrapolate a sinusoid using various k-step ahead models. Main takeaway is that errors accumulate when this happens. Refers to Chapter 8.1 of d2l.ai. 
`text_models.ipynb`: Refers to Chapters 8.2-3 of d2l.ai. 

