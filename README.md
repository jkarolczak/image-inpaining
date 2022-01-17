# Image inpainting using GAN

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
*Debugging stops logging to neptune and display intermediate results.*
> `python train.py --debug`



