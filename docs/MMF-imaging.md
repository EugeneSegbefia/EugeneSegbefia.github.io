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

* [The Phase conjunction method](https://opg.optica.org/ol/abstract.cfm?URI=ol-7-11-558):
    Reversing wavefront distortions by capturing and conjugating the phase of the   transmitted light.

* [The Matrix Recreation Method](https://doi.org/10.1038/ncomms1078):
    Reconstructing the fiber's transmission matrix by analyzing known input-output relationships.

* [Constructing the complete transmission matrix of the fiber using intensity data](https://doi.org/10.1038/nphoton.2015.112):
    Inferring the complete transmission matrix using only intensity measurements, avoiding direct phase retrieval.
  
* [*_Throwing the problem at a neural network_*](https://doi.org/10.1364/OL.16.000645):
    Training a neural network to learn the fiber’s transformation and reconstruct the input from the output.

## The Neural Network Approach 
The approach to solving this problem that was focused on in this project is the approach that involves training a neural network. There have been various attempts by researchers to successfully transmit image information using different neural network architectures. Some work that directly influenced and inspired my work are: 

* Single Dense Hidden Layer Network - [Tom Kuusela](https://doi.org/10.1119/5.0102369)
* Single Dense Hidden Layer Network vs Convolutional Neural Network - [Changyan Zhu et al](https://doi.org/10.1038/s41598-020-79646-8) 
* Convolutional Neural Network - [Babak Rahmani et al](https://doi.org/10.1038/s41377-018-0074-1)
* Single Complex Dense Neural Network - [Piergiorio Caramazza et al](https://doi.org/10.1038/s41467-019-10057-8)

These works served as guides for me while I explored the problem and started working on making a practical solution to the transmission problem explained above. After careful reading and a bit of experimentation, the model I felt held the most promise was the one proposed by Caramazza et al. 


## Data Used
Image Data was used for the training and testing of the models described in this experiment where the image data  was collected and published by Caramazza et al. The database provided contained *50,000* image pairs in the training dataset. I am referring to an input image and its corresponding output speckle pattern image. The test dataset provided contained 1,000 image pairs.


## Model Used 
The model proposed by Caramazza et al which I implemented in this project is fairly rudimentary, all neural network architectures published considered. It consists of an input layer, a complex dense hidden layer, an amplitude layer, and an output layer.  
A Dense hidden layer is undeniably the most used layer in neural network design and creation regardless of the task the model is intended to work on. This architecture involved redesigning the commonly used dense layer to work with complex numbers and in that way reconstruct the transmission matrix of the multimode fiber cable used  which is complex-valued. 

```python
def create_complex_model():
    input_img = Input(shape=(image_dim*image_dim, 2))
    l = input_img
    l = ComplexDense(orig_dim*orig_dim, use_bias=False, kernel_regularizer=regularizers.l2(lamb))(l)
    l = Amplitude()(l)
    out_layer = l
    model = Model(inputs=input_img, outputs=[out_layer])
    return model 
```


### Model Details 
#### Input Layer 
For this model, the inputs were the image pairs as they are tensors containing all the pixel information. there number of neurons was equivalent to the number of pixels of the input image of the image pairs.

#### Complex Hidden Layer 
This layer as mentioned earlier is a spin-off of the complex dense layer, so with the help of some helper functions a complex dense layer was created with the sole purpose of having the neuron values being used be complex numbers.

```python
class ComplexDense(Layer):

    def __init__(self, output_dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        super(ComplexDense, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)


    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim, 2),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      #constraint=self.kernel_constraint,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.output_dim, 2),
                                        initializer=self.bias_initializer,
                                        #regularizer=self.bias_regularizer,
                                        #constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None
        super(ComplexDense, self).build(input_shape)

    def call(self, X):
        # True Complex Multiplication (by channel combination)
        complex_X = channels_to_complex(X)
        complex_W = channels_to_complex(self.kernel)

        complex_res = complex_X @ complex_W

        if self.use_bias:
            complex_b = channels_to_complex(self.bias)
            complex_res = K.bias_add(complex_res, complex_b)

        output = complex_to_channels(complex_res)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, 2)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            #'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            #'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            #'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            #'kernel_constraint': constraints.serialize(self.kernel_constraint),
            #'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

#### Amplitude Layer 
In this layer, we convert the predicted complex numbers back into real numbers and compare the predicted image data to the target image data it does this by finding the magnitude of the complex number.

```python
class Amplitude(Layer):

    def __init__(self, **kwargs):
        super(Amplitude, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Amplitude, self).build(input_shape)

    def call(self, X):
        complex_X = channels_to_complex(X)
        output = tf.abs(complex_X)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])
```
#### Output Layer 
Here we output the magnitudes of the complex numbers we predict the pixels to have.


### Training 
A key point of this experiment was to not only build a working dense hidden layer neural network model but it was to explore the effects of different loss functions on the observed predicted images. In this project we observe the effects  of these loss functions: 

* L1 loss    
* L2 loss
* SSIM loss
* SSIM loss + L1 Loss

The results of the models were compiled and compared for uniformity all models were trained using the same hyperparameters:

```python
image_dim = 120
orig_dim = 92
length_image = 10000
epochs = 850
lr = 1e-5
batch_size_n = 32
lamb = 0.1
```
The other model specifications that are worth nothing are the optimizer used and the reduce learning rate function used in the training of the model. 10,000 image pairs were used in the training of the model. 

```python
model.compile(optimizer=SGD(learning_rate=lr), loss=loss_function, metrics=['accuracy'])
    
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=2, min_lr=lr / 1e3, verbose=1
    )
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[reduce_lr],
    )
```

## Results 
We observe how well the models recreate the input images and compare the fidelity of the images recreated by each model.

The values of these loss functions were the only metric we monitored strictly 
![Loss Progression](https://EugeneSegbefia.github.io/figures/losstracked.png)  

[**Figure 2: Loss Progression**](#fig-2){:id="fig-2"}  
*Loss progression of the models over training epochs.*

| Loss Function | Visualization |
|--------------|--------------|
| **SSIM Loss** | ![SSIM Loss](https://EugeneSegbefia.github.io/figures/SSIMloss.png) |
| **L2 Loss** | ![L2 Loss](https://EugeneSegbefia.github.io/figures/l2loss.png) |
| **L1 Loss** | ![L1 Loss](https://EugeneSegbefia.github.io/figures/L1loss.png) |
| **SSIM + L1 Loss** | ![SSIM + L1 Loss](https://EugeneSegbefia.github.io/figures/SSIMlossl1loss.png) |

[**Figure 3: Model Comparison**](#fig-1){:id="fig-1"}  
*A comparison of different loss functions experimented on.*  

We observe the clearest results with the model trained with the loss that is a combination if the SSIM loss and the L1 loss.

## Discussion 

While decent results were observed it would be worthwhile to experiment a bit more with the model used such as viewing the effect of adding multiple of these dense hidden layers and observing how it would affect the quality of image recreation. 

It would also be worthwhile to observe the effects of training with a large dataset possibly the entire training data. This would be ideal and possible provided I have access to much stronger computational resources


## Conclusion 

Projects done by myself and other members of the scientific community show that there is promise in the use of multimode fiber cables and it is worthwhile to pursue methods to regularize its use in our day-to-day lives. There is more work to be than and this a field of study where there is a lot of room for growth for the improvement and development of the field.




