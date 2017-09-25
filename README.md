#Lung Nodule Detection in CT Scans Images#



Dataset: https://luna16.grand-challenge.org

###Which does what?###

**create_2d_patches.py**: Creates 2D patches of specific size around nodule coordinates(World coordinates).

**create_2d_images.py**: Creates 2D full image(512x512) containing nodules in 3D scan images.

**resnet_finetune_image.py**: Train full images(downscaled to 224x224) on a Resnet model which was previously trained on X-ray images.

**resnet_finetune_patch.py**: Train 2D patches of smaller size(say 100) on a Resnet model which was previously trained on X-ray images.

**myscript_tamjid.slurm**: SLurm script to run the code in NERSC server.
