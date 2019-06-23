r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.8
    reg = 1
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


1. Surprisingly, the best results were yielded by the most shallow network (L2).
This corresponds with with the fact that networks deeper than 4 layers were not trainable.
In larger depth our model could not learn.

This is probably due to the large amount of max pooling layers. Any region not contributing to the maximum value,
Looses it's affect on the gradient.

As there is a large number of such layer, it becomes very probable that values in the first layers of the network will
Loose their affect on the gradient and thus will not learn. Another factor that might harm is a form of vanishing gradient:
Though not as bad as in the case with the sigmoid function, dead ReLu's can also kill off gradients
(all negative values are set by zero). When the network is deep, this can cause the higher layers
to learn very slowly or as in our case- die off completely.

Without addressing this problem, the more shallow networks have a bigger learning potential and thus yield better results.

2. Yes, networks deeper than 4 did not learn.
   As said above this is due to to pooling layers or dead ReLus.
   Ways to address this problem is to use:
        1. Use dropout (or skip connections as in Res-Nets) to give all values a chance to affect the gradient.
        2. Use Leaky ReLu's to avoid killing non linearity.
            
"""


part3_q2 = r"""
**Your answer:**


Test 1.2 yielded better results as we added more convolution filters which improved our feature extraction.
In this test the better preformance was yielded by the 4 layer network with the 256 conv filters.
The larger amount of filters added in two ways: it inhanced the feature extraction and since pooling layers don't affect
the channel dimension it made our network train a bit better in deeper architectures, but again the 8 layer network did
still could not learn.

"""

part3_q3 = r"""
**Your answer:**

As the additional conv filters had effectively stretched the network, now the networks with 3 and 4 layers did not train.
Both the 1 and 2 layer networks yielded improved results over the networks in exp 1 and 2, which used only one
convolution depth over the entire network. The best results were yielded by the 2 layer deep network, as it had more
trainable filters than the shallower one. The improved results are probably due to the fact that the varying conv depth
improved the networks ability to learn more complicated patterns.  

"""


part3_q4 = r"""
**Your answer:**

The changes we made to our model are:
    1. Adding dropout layers.
    2. Adding batch normalization layers.
    We also tried changing the activation function to Leaky-ReLu but it yielded less than optimal results.
    

The first thing noticeable between our module and the previous ones is that it was able to train also deeper
networks thanks to the dropout layers which allowed gradient flow to the first layers.

Secondly, the improved model learned much faster than previous ones and yielded better accuracy, the best configuration
was the four layer one maxing at about 85% test accuracy. We asses that again, dropout played a major role in this 
improvement as it allowed more gradient flow and made the network classify using multiple features (as at any pass
some of the features may be dropped out). 

"""

# ==============
# Part 4 answers


def part4_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============


# ==============
# Part 5 answers

PART5_CUSTOM_DATA_URL = None


def part5_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 6 answers

PART6_CUSTOM_DATA_URL = None


def part6_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['z_dim'] = 128
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.2
    
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.0002
    hypers['discriminator_optimizer']['weight_decay'] = 0.02
    hypers['discriminator_optimizer']['betas'] = (0.5, 0.999)
    
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['weight_decay'] = 0.02
    hypers['generator_optimizer']['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part6_q1 = r"""
**Your answer:**

The loss of the generator as we defined it, is basically: "how much did we manage to fool the discriminator?"

Therefore, the gradients of the generators loss function are dependent on gradients calculated for the discriminators 
loss function.

When we train the discriminator, we obviously don't want it to affect the training of the generator, and vice versa.

But, since for every batch (or several batches, up to us) we first train the discriminator and then the generator, 

when we sample from the generator in order to train the discriminator, **we will need to discard the the gradients**.

Otherwise, the training of the discriminator will impact the training of the generator!

"""

part6_q2 = r"""
**Your answer:**

**1.**

No! the generators loss is not the only measurement of success: if our discriminator isn't doing a good job at 
separating fake images from real images, the generator is not necessarily generating good images, but rather just doing
whatever is necessary in order to full our discriminator.

Therefore, we should stop the training only when **both losses** - the generators and the discriminators - are low. 

**2.**

It means that while we may not see any change in the discriminators loss- the discriminator keeps learning:

Since we know our generators loss is getting lower - which means the generator is getting better at fulling the 
discriminator, if our discriminator had stopped learning, then its loss should be going **down** and not staying stable.

However, it is possible that our discriminator is not getting any better "separating" real and fake images, but 
rather just at identifying real images as real: improving the classification on real images and decreasing the 
classification on fake images. This could explain why the loss stays stable.

"""

part6_q3 = r"""
**Your answer:**

Not mandatory :)

"""
