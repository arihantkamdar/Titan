# Titan
A simple implementation of TITANs Paper by Google

In intesrest of time, I was just able to implemet MAC architecture of the paper, However in next couple of days I would work on MAG and MAL architectures irrespective of the decision of the hiring. I just want to thank you as this assesment helped me work on this that I was wanting to do for so long. 

# MAC: Memory-As-Context Model Implementation

This repository contains a PyTorch implementation of the **Memory-As-Context (MAC)** architecture from the TITANS paper. MAC is a transformer-based model enhanced with persistent and long-term memory modules that adapt at test time to improve sequence modeling tasks.

---

## Overview

The MAC model integrates three types of memory:

- **Persistent Memory:** Fixed learned vectors that provide constant contextual information.
- **Long-Term Memory:** A trainable MLP module that processes the input embeddings and attention outputs.
- **Short-Term Memory:** Implemented via a standard Multi-Head Attention mechanism over chunks (segments) of the input sequence combined with persistent and long-term memory.

The model processes long sequences by chunking them into manageable segments, applying attention with causal masking, and updating its memory based on reconstruction losses at both training and test time.

Test-time adaptation is achieved by updating the long-term memory module via gradient descent steps on memory reconstruction losses, without affecting the main model parameters.

---

## Features

- Chunked sequence processing for efficient long sequence modeling: The paper argues that this could be done in parallel and thus GPU accelearation can be used, but for the sake of simplicity I processed them sequentially
- Separate optimizers for main model parameters and memory module to enable test-time adaptation : Since suprise learning has Momentum and Decay , I thought the it would be nice to implement SGD
- Memory update steps implemented as a gradient-based adaptation mechanism during inference: Noe this just happen on a single sequence and them is reset after exemplare is processed
- Combination of task loss (MSE) and memory loss to jointly train the model.
- Example training and evaluation routines on synthetic data for quick experimentation.

---

## Installation

Requires Python 3.8+ and PyTorch.

```bash
pip install torch
