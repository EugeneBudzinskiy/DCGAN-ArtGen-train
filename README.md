# DCGAN Art Generation (Train Module)


## Description


This is the training module for the image generation project based on Deep Convolutional 
Generative Adversarial Network (DCGAN). This module is responsible for dataset processing, 
models training process, training progress logging and trained models extraction. 

The creation of this project was motivated by the theoretical potential to reduce the 
human resources needed to create art. The main goal of this project is to simplify and 
algorithmize the process of image creation.

Additionally, this project was a form of self-founded experience to understand how 
Generative Models work, are created and trained. Specifically was chosen Generative 
Adversarial Networks (GANs) because they are relatively easy and fast to train. 

This project can help humans to make the process of art creation faster by providing 
some form of inspiration or even partially replacing human-created art.

In the end, it became clear that GANs are not that easy to train, and they are pretty 
unstable during training. The main problem for the training process was 
['mode collapse'](https://arxiv.org/abs/1406.2661). To battle this issue was implemented 
a lot of different modifications. The most successful was 
['WGAN-GP'](https://arxiv.org/abs/1704.00028) which helps with 'mode collapse' but at 
the cost of training time. In general, the FID score for the 'WGAN-GP' model was 128.259, 
which can be classified as a satisfactory result.


## Installation


Implementation was written using Python 3.9 

In file [requirements.txt](requirements.txt) are listed main libraries which are necessary 
for this project. For installing requirements run this command (inside project folder): 

	pip install -r requirements.txt

For starting of training process also required to download dataset 
[ArtBench-10](https://github.com/liaopeiyuan/artbench). In project root exist three folder 
for dataset ([dataset_lsun](dataset_lsun), [dataset_lsun_unpacked](dataset_lsun_unpacked), 
[dataset_memmap](dataset_memmap)). Each folder corresponds to same daset but in different 
stages of dataset preprocessing process. 

Folder [dataset_lsun](dataset_lsun) corresponds to data packed in 
[LSUN](https://arxiv.org/abs/1506.03365) format which can be downloaded from this 
[link](https://drive.google.com/drive/folders/1gWdbot6wfmvsI1UDY8WC_-vkZsK9VEhM?usp=sharing) 
(Warning! Images in original size, approximately 50GB).

Folder [dataset_lsun_unpacked](dataset_lsun_unpacked) corresponds to unpacked data from 
[dataset_lsun](dataset_lsun). Unpacking process using original implementation for LSUN 
datasets ([link](https://github.com/fyu/lsun)). Code fragment which are has been used
can be found in [lsun](lsun) folder.

Folder [dataset_memmap](dataset_memmap) corresponds to preprocessed dataset from folder 
[dataset_lsun_unpacked](dataset_lsun_unpacked). On this step of preprocessing unpacked
images are crop to be squares-like and then downscaled to 64x64 pixels. Result are saved
in numpy memmap format.
   

## Usage


1. Download [ArtBench-10](https://github.com/liaopeiyuan/artbench) original size, LSUN 
encoded, dataset to [dataset_lsun](dataset_lsun) folder
2. To convert dataset to correct format run script [`convert.py`](convert.py)
3. To start train process run script [`train.py`](train.py)
4. Wait until end of training process (may take a long time)
5. To extract pretrained model run scropt [`extract.py`](extract.py) (model
will be saved to [xtr_generators](xtr_generators) folder)
6. End of training process. Use extracted pretrained model in other module
    

## Credits


* [Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., 
Ozair, S., Courville, A., & Bengio, Y. (2014, June 10). Generative Adversarial 
Networks](https://arxiv.org/abs/1406.2661)
* [Goodfellow, I. (2017, April 3). NIPS 2016 tutorial: Generative Adversarial 
Networks](https://arxiv.org/abs/1701.00160)
* [Brownlee, J. (2019, July 19). A gentle introduction to generative adversarial 
networks (Gans)](
https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)
* [Radford, A., Metz, L., & Chintala, S. (2016, January 7). Unsupervised
representation learning with deep convolutional generative Adversarial 
Networks](https://arxiv.org/abs/1511.06434)
* [Liao, P., Li, X., Liu, X., & Keutzer, K. (2022, June 22). The ARTBENCH 
dataset: Benchmarking generative models with artworks](
https://arxiv.org/abs/2206.11404)
* [Yu, F., Seff, A., Zhang, Y., Song, S., Funkhouser, T., & Xiao, J. (2021).
Fyu/LSUN: LSUN dataset documentation and demo code](https://github.com/fyu/lsun)
* [Yu, F., Seff, A., Zhang, Y., Song, S., Funkhouser, T., & Xiao, J. (2016, 
June 4). LSUN: Construction of a large-scale image dataset using Deep Learning 
with humans in the loop](https://arxiv.org/abs/1506.03365)
* [Borji, A. (2018, October 24). Pros and cons of gan evaluation measures](
https://arxiv.org/abs/1802.03446)
* [Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2015, 
December 11). Rethinking the inception architecture for computer vision](
https://arxiv.org/abs/1512.00567)
* [Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. 
(2018, January 12). Gans trained by a two time-scale update rule converge 
to a local Nash equilibrium](https://arxiv.org/abs/1706.08500)
* [Brownlee, J. (2019, October 10). How to implement the Frechet Inception
Distance (FID) for evaluating Gans](
https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/)
* [Fernandez, D. L. (2022, May 17). Gan convergence and stability: Eight
techniques explained](
https://davidleonfdez.github.io/gan/2022/05/17/gan-convergence-stability.html)
* [Arjovsky, M., Chintala, S., & Bottou, L. (2017, December 6). Wasserstein 
Gan](https://arxiv.org/abs/1701.07875)
* [Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. 
(2017, December 25). Improved training of Wasserstein Gans](
https://arxiv.org/abs/1704.00028)


## License


Licensed under the [MIT](LICENSE) license.
