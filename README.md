# black-box-attacks
Comparison of gradient estimation techniques for black-box adversarial examples

This is a fork of https://github.com/labsix/limited-blackbox-attacks

Choices of gradient estimation are: `NES`, `RDSA`, `SPSA`, `SPSA (1-sided)`

To run:


1. Download Make a directory tools/data, and in it put the decompressed Inceptionv3  
classifier from (http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)


2. Set IMAGENET_PATH and METADATA_PATH in main.py and attacks.py to the location of the 
ImageNet dataset on your machine.

3. To run all experiments: `./query.sh`
