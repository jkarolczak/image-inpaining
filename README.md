# Image inpainting using GAN

## Report
A report and live demo are available [here](https://share.streamlit.io/jkarolczak/image-inpainting/main/report/app.py)

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

To performe inference using generator run:
> `python infer.py --statedict path_to_statedict --images 10`
Where `--path_to_statedict` stands for file to a pickled generators state dict and `--images` stands for number of images to use. Specifying number of images may be omitted.