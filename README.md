# Image inpainting using GAN

## Configuration and usage

To create environment using Conda and `yaml` files:
> `conda env create -f environment.yaml`

To activate created environment:
> `conda activate image-inpainting`

To parametrize network or training:
> Edit `yaml` adequate file in the `/cfg` directory.

To integrate training with neptune:<br>
*For more details see: https://docs.neptune.ai/getting-started/hello-world*
> Create `/src/neptune.yaml` file containing project name and token using temaplate in the file `/src/neptune_template.yaml`

To run the training process:
> `python train.py`

To run the training process in debugging mode:<br>
*Debugging stops logging to neptune and display intermediate results to standard output.*
> `python train.py --debug`

## Theory behind

Used notation:
- `netG` - neural network acting as a generator 
- `netGD` - neural network acting as a global discriminator evaluating whether the whole image seems to be real 
- `netLD` - neural network acting as a local discriminator evaluating whether the erased area seems to be real

Computations are done in two phases:
- stage 1 - the generator (`netG`) is fitted in a casual manner using a loss and an optimizer. This significantly reduce time and space complexity. Hence this step provide generator's initial weights in a reasonable time.
- stage 2 - generator is fine-tuned using global (`netGD`) and local (`netLD`) discriminators. This stage follows GAN architecture.