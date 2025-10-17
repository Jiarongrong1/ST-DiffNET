# ST-DiffNet: Diffusion-Based Inpainting of Dunhuang Murals with Structural and Textural Guidance

Diffusion models work by progressively adding Gaussian noise to destroy training data and then learning to recover the data by reversing this noising process. After training, we can use the diffusion model to generate data by simply passing randomly sampled noise through the learned denoising process. In this project, we extend this idea to the restoration of Dunhuang mural images.

# Dataset

We use the real dunhuang dataset to train and evaluate our method. To gain access to this dataset, please contact the author via email for specific details, as it is subject to private permissions requirements.

# Code
## Requirement

``pip install -r requirement.txt``

## Training on Dunhuang Murals Dataset

We are training on the Dunhuang mural dataset. Modify train.py to set img_file, mask_file, and structure_file, and create the corresponding folders. Then, run the following script:
python train.py

## Testing

Modify sample.py to set img_file, mask_file, and structure_file, and create the corresponding folders. Then, run the following script:
python sample.py
