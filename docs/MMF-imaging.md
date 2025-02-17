layout: page
title: "Multimode Fiber Imaging"
permalink: /MMF-imaging

# Imaging through a multimode fiber cable 
## Motivation
Single-mode optical fiber cables are used universally for our day-to-day telecommunication needs. A single-mode optical fiber cable allows a single beam of  light to pass through the cable at a time this means we can only encode one pixel of image information on a beam to be transmitted through a cable at a time. This is unfortunately quite inefficient and means that getting one image from one location to another would require using large numbers of these cables. To find a more efficient approach to solving this problem a multimode fiber cable was designed and created the key difference being that it would be able to have multiple "modes" of light pass through it at once. Unfortunately, when we try to do this, pass multiple  beams of light with pixel information we observe mode and phase mixing this is when we observe the electromagnetic fields of these beams interfering and influencing each other. This results in the output image being observed as a speckle pattern image.


<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="https://EugeneSegbefia.github.io/figures/Speckle_pattern.png" alt="Speckle Image" style="width: 45%;">
  <img src="https://EugeneSegbefia.github.io/figures/originalImage.png" alt="Original Image" style="width: 45%;">
</div>

[**Speckle vs Input Images**](#fig-1){:id="fig-1"}  
*A comparison between a possible input image and its corresponding output.*

## Attempts to Solve this Problem 
In the scientific community, there have been multiple attempts to solve this problem, with one of these attempts or methods being the focus of this project the said methods are.

* The Phase conjunction method:
    Reversing wavefront distortions by capturing and conjugating the phase of the   transmitted light.

* The Matrix Recreation Method:
    Reconstructing the fiber's transmission matrix by analyzing known input-output relationships.

* Constructing the complete transmission matrix of the fiber using intensity data.
    Inferring the complete transmission matrix using only intensity measurements, avoiding direct phase retrieval.
  
* _*Throwing the problem at a neural network*_.
    Training a neural network to learn the fiber’s transformation and reconstruct the input from the output.

## The Neural Network Approach 
The approach to solving this problem that was focused on in this project is the approach that involves training a neural network. There have been various attempts by researchers to successfully transmit image information using different neural network architectures. Some work that directly influenced and inspired my work are: 

* Single Dense Hidden Layer Network - Tom Kuusela
* Single Dense Hidden Layer Network vs Convolutional Neural Network - Changyan Zhu et al 
* Convolutional Neural Network - Babak Rahmani et al
* Single Complex Dense Neural Network - Piergiorio Caramazza et al


These works served as guides for me while I explored the problem and started working on making a practical solution to the transmission problem explained above. After careful reading and a bit of experimentation, the model I landed on that I felt held the most promise was the one proposed by Caramazza et al. This architecture involved redesigning the commonly used dense layer to work with complex numbers and in that way reconstruct the transmission matrix of the multimode fiber cable used  which is complex-valued.


## Data Used
Image Data was used for the training and testing of the models described in this experiment where the image data  was collected and published by Caramazza et al. The database provided  






