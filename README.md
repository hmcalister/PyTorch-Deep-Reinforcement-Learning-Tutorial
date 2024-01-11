# Pytorch Deep Q Network Tutorial

This repository implements the [Pytorch Deep Q Network tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) and any additional playing around with the tutorial I happen to do. This is not my own idea, it is largely taken from the tutorial. The aim of this project is:

- Familiarize myself with pytorch (coming from using exclusively tensorflow in the past)
- Have a look at reinforcement learning in a neural network context
- Work on a deep Q network problem

## Running this Project

Ensure you have [set up and installed PyTorch](https://pytorch.org), preferably in a new [Conda](https://docs.conda.io/en/latest/miniconda.html) environment. Ensure you have also installed the following libraries with `pip`:

```bash
pip3 install numpy matplotlib 
pip3 install torch torchvision torchaudio
pip3 install 'gymnasium[classic_control]'
```

You may also want to ensure you have cuda set up and installed on your machine, as PyTorch can hook into cuda (quite easily, may I add!) to drastically improve training times. However, I offer you no links to do this, as it really is a quite individual experience, and what works for my Fedora Linux machine will be totally different to your Ubuntu machine, different again for your Windows machine, and will certainly not work on your MacOS machine!

### Further notes.

While this tutorial is now complete and in working order, I would still like to understand what exactly is going on in it a little better! I suppose I will spend a little while poking around the various PyTorch functions, adding comments here and there as I find out.