# Workspace for Implementing Light Classification

## Folders

- config: configuration files for training a network using TensorFlow
- data-lechner: datasets from Alex Lechner
- data-vatsal: datasets from Mr. Vatsal. I used this dataset to train.
- models: TensorFlow Models
- pre-trained-model: ssd_inception_v2_coco_2018_01_28. This can be found from TensorFlow Models zoo.
- test_images_sim: some examples from the simulated camera in the simulator
- test_images_site: some examples form the Udacity parking lot using the real car, Carla.

## Create TFRecord files

Note: Due to ROS's dependency to Python 2.7, I have to use 2.7 for the whole system. But this part can be done separately. So I used Python 3.6 for this part.

To train a model with the TensorFlow, we have to convert Vatsal Srivastava's data to the TFRecord format. (Note that Alex Lechner's dataset is conveniently in the TFRecord format)

Change directory to `models/research`. Then run the followings.
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

```
python ../../data_conversion_udacity_sim.py --output_path ../../data-vatsal/sim_training_data.record
```
```
python ../../data_conversion_udacity_site.py --output_path ../../data-vatsal/site_training_data.record
```
These two script will generate two TFRecord files: `sim_training_data.record` and `site_training_data.record`.
