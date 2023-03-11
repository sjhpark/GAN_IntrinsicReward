# Generative Adversarial Network (GAN) with Intrinsic Reward

### DCGAN with a simple intrinsic reward implemention for novel image generation

---
## Abstract
This project implements a Generative Adversarial Network (GAN) with Intrinsic Reward, 
which is a modification of the traditional GAN model. 
The intrinsic reward provides a reward signal to the generator during training, in addition to the adversarial loss, 
and is designed to encourage the generator to produce diverse and novel images.

This project was an experiment to implement a simple intrinsic reward model to a generative AI model 
for image generation to see if it can guide the generator to novel image generation.

## GAN
The GAN model is trained using PyTorch, and it consists of a generator and a discriminator. 
The generator generates fake images from random noise, while the discriminator tries to distinguish between real and fake images. 
During training, the generator tries to improve its ability to generate realistic images, 
while the discriminator tries to improve its ability to distinguish between real and fake images.

## Intrinsic Reward
The intrinsic reward was used to guide the generator's learning process in addition to the usual adversarial loss. 
After generating fake images from random noise, the generator's output was passed through the intrinsic reward model to compute the intrinsic reward loss. 
This loss was then added to the generator loss to obtain the total loss used to update the generator's parameters. 
The intrinsic reward loss is designed to encourage the generator to produce images that contain certain desirable features 
or qualities beyond simply being realistic-looking. 
In this implementation, the intrinsic reward model's output was compared to the original random noise used to generate the fake images, 
and the difference (L2 loss or squared L2 norm) between these was used as the intrinsic reward loss.

$$loss_{IR} = ||IR(G(z)) - z||_2^2$$

## Dataset
An __art portrait__ subset of __Wiki-Art: Visual Art Encyclopedia__ dataset was used to train the GAN model. This subset contains 4,117 art portrait images.

The dataset was found from the links below:

__Wiki-Art: Visual Art Encyclopedia__: https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan

An __art portrait__ subset: https://www.kaggle.com/datasets/karnikakapoor/art-portraits

![image](https://user-images.githubusercontent.com/83327791/224470564-8b4f739c-ad4c-4f3a-84ac-b6d477afff23.png)

## Preprocessing
The training dataset, consisting of art portraits, was preprocessed to reduce the image size to 64 x 64 for faster training and better memory efficiency. This was done using the PyTorch transforms module, which applies a series of image transformations to the dataset.

First, the images were resized to the desired size using transforms.Resize(img_size). Then, a center crop of the same size was taken using transforms.CenterCrop(img_size) to ensure that all images are of the same size. To increase the diversity of the training dataset, transforms.RandomHorizontalFlip(p=0.5) was applied to randomly flip the images horizontally with a probability of 0.5.

To further augment the dataset, random color jitter and rotation were applied to the images using transforms.ColorJitter() and transforms.RandomRotation(degrees=20). These transforms were randomly applied to each image with a probability of 0.2 using transforms.RandomApply(random_transforms, p=0.2).

After the image transforms, the images were converted to PyTorch tensors using transforms.ToTensor(), and then normalized using transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)). This ensures that the pixel values are in the range of [-1, 1], which is suitable for training GANs.

The preprocessed dataset was then loaded into a PyTorch DataLoader with a specified batch size of 32, using DataLoader(train_data, batch_size=batch_size, shuffle=True). The shuffle=True argument ensures that the images are randomly shuffled before being loaded into each batch during training.

Finally, to check that the preprocessing was done correctly, a batch of images and their corresponding labels were extracted from the train_loader using imgs, label = next(iter(train_loader)). The images were then transposed to have a shape of (batch_size, height, width, channels) using imgs = imgs.numpy().transpose(0, 2, 3, 1).

![image](https://user-images.githubusercontent.com/83327791/224470783-c164bee6-4c6d-4933-99d4-9200f33cbe7d.png)

## Demonstration
"To compare the generated images of the GAN without intrinsic reward and the GAN with intrinsic reward, 
I set the random seed as 3407 and applied weight initialization to the generator, discriminator, and intrinsic reward networks. 
This ensured that both models used the same batch of real input images and a fixed (latent) noise vector for generating testing images.

During the training process, I used a 128-dimensional latent vector as input to the generator. 
This means there are 128 scalar values (noise) that the generator can adjust to produce different outputs. 
The larger the dimensionality of the latent space, the more complex outputs can be generated, 
but the training process would be slower.

__GAN without Intrinsic Reward after 50 epochs of training:__

![generated_images_no_intrinsic_reward](https://user-images.githubusercontent.com/83327791/224469386-479cc59b-b37d-4848-a8f8-dcfbccf39301.gif)

![image](https://user-images.githubusercontent.com/83327791/224470885-a5c85be6-3cd4-4733-ab5e-0ac2c81e7200.png)

__GAN with Intrinsic Reward after 50 epochs of training__

![generated_images_intrinsic_reward](https://user-images.githubusercontent.com/83327791/224469391-b2110d9d-a0ed-4b46-b0eb-b93649f859ed.gif)

![image](https://user-images.githubusercontent.com/83327791/224470868-23777437-a551-4483-be45-1848d225d10b.png)

## Challenge
It is hard to compare the novelty of two set of images. 
Therefore, further research is needed to explore metrics for novelty comparison.

Some possible metrics are:
- Wasserstein Distance
- Entropy Score
- Information Gain (not a metric)
