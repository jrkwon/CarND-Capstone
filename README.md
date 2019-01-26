# CarND-Capstone

## Docker

I chose the use Docker to complete the project. There exist problems here and there to stick to the Docker container. But this would save my time in the long-run since I do not need to worry about dependencies between packages.


### Run a container 

Port 4567 is used to communicate ROS nodes with the simulator, we must map the port number. My local current directory will be mounted to `/capstone`. So make sure you are at the folder that you cloned from the Udacity repository `CarND-Capstone`.


```
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --name jaerock_capstone --rm -it capstone_with_cuda9
```

### Multiple terminals

```
docker exec -it jaerock_capstone bash
```

## Traffic Light Detector

My initial choice was OpenCV with YOLO3. But the current OpenCV implementation only support CPU and some GPU from Intel. I tested out the CPU version but it turned out too much time to have a detection result even though OpenCV argues that its CPU version is several times faster than the Darknet's CPU version. So I had to rule this option out. The second choice was to use the Darknet's YOLO3 GPU version, the original implementation. But some dependent software modules require Python 3+. But I have to stick to Python 2.7 due to the ROS's dependency on it. Thus, this option was also ruled out. There exist some Python wrapper of YOLO3 but they also require Python 3+. After the study about the possible options for object detection, I decided to use the object detection feature of TensorFlow to detect traffic lights to implement traffic light classification. There were some issues in using Python 2.7 with Object Detection API of TensorFlow but they were manageable. 

### Install Software

Here, I explain how to update `requirements.txt` and `Dockerfile` to install the required software packages. 

#### OpenCV

Add `opencv-python==3.4.2` to the `requirements.txt`. After running the docker container if you have not started, run the `pip` command as below to install the OpenCV.

#### TensorFlow GPU

Replace `tensorflow=1.3` line with `tensorflow-gpu==1.12.0` in the `requirements.txt`.


#### CUDA 9.0 with cuDNN 7.0

We need to nstall CUDA 9.0 with cuDNN 7.0 to the current docker container to use the object detection feature of TensorFlow. The code below must be added into `Dockerfile`.

```
# CUDA 9.0-base and runtime >>>>>>>>>>>>>>>>>>>>
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
    rm -rf /var/lib/apt/lists/* && \
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDA_VERSION 9.0.176

ENV CUDA_PKG_VERSION 9-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

ENV NCCL_VERSION 2.3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-cublas-9-0=9.0.176.4-1 \
        libnccl2=$NCCL_VERSION-1+cuda9.0 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDNN_VERSION 7.4.1.5
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

```

### Create a Docker image with CUDA 9.0

Now, it is time to build a new Docker image that has the required software packages.

```
sudo docker build . -t capstone_with_cuda9
```

Note that sudo is required due to a file under the folder, `models/research/bin`.

### Training

Create a folder named `light_classification` that will be the main folder of the training.

I will use a _transfer learning_ that utilizes a pre-trained model for image classification, _add_ more data to train, in my case, they are traffic lights, and change the what to classify and the number of classes. 


#### Datasets


I downloaded datasets from 
- [Vatsal Srivastava's dataset, originally shared by Anthony Sarkis](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing))
- [Alex Lechner's dataset](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0)

And save them into the folder that has been created.
Make folder names as `data-vatsal` and `data-lechner` respectively inside the folder, `light_classification`.

Note: I changed `real` to `site` in a folder and a yaml file for the site data to maintain consistency with source code reference. The source code refers to the Udacity parking lot as _site_ instead of _real_.


#### TensorFlow Models

Change directory to `light_classification` if you are not in it.

I downloaded TensorFlow Models using `git clone`

```
git clone https://github.com/tensorflow/models.git
```
You will see `models` folder now.

#### Create TFRecord Files

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

#### Pre-trained Model

`SSD Inception V2 Coco` was selected because it shows a balanced output in terms of the speed and accuracy according to other Udacity students.

Download the pre-trained model from [the model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Find `ssd_inception_v2_coco` and download it.

Change directory to the main folder, `light_classification` and create a folder named `pre-trained-model`. I untarred the tar.gz file into the folder.

```
tar -xvzf ~/Download/ssd_inception_v2_coco_2018_01_28.tar.gz
```

#### Create Config Files

Two files were created: `ssd_inception_v2_coco_traffic_light_sim.config` and `ssd_inception_v2_coco_traffic_light_site.config`. They are for simulation images and Udacity parking lot images respectively to use fine tuning with the `ssd_inception_v2_coco` model

I set `num_steps` to 5,000 for the traffic light images that are clear and simple. The `num_step` for the Udacity parking lot images was set to 10,000 since the quality of traffic light images was not good.

I did not use evaluation data due to the simplicity of the classification task.

#### TensorFlow Object Detection API dependencies

TensorFlow Object Detection API depends on some libraries that are not part of TensorFlow. 

[Install TensorFlow Object Detection API dependencies](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

#### Start Training

From the main `light_classification` folder, run the followings.

```
python models/research/object_detection/legacy/train.py --pipeline_config_path=config/ssd_inception_v2_coco_traffic_light_sim.config --train_dir=data-vatsal/sim_training_data

```

```
python models/research/object_detection/legacy/train.py --pipeline_config_path=config/ssd_inception_v2_coco_traffic_light_site.config --train_dir=data-vatsal/site_training_data

```

#### Freeze Graph

Using `export_inference_graph.py` under `object_detection`, we can get a frozen inference graph, `frozen_inference_graph.pb`.

```
python models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path=config/ssd_inception_v2_coco_traffic_light_sim.config --trained_checkpoint_prefix data-vatsal/sim_training_data/model.ckpt-5000 --output_directory data-vatsal/sim_training_data
```

#### Modification of TLClassifier

The original TLClassifier does not have any argument on its constructor. I added one boolean value to indicate that the ROS node runs inside the Simulator or the real car, Carla by using the `is_site` param. This makes me to selectively load a trained model.

#### Handling Lags

After implementing the classifier, some errors from `waypoint_updater` and `waypoints_2d`. This is because of the much longer delay due to the inference engine of Object Detection APIs of TensorFlow.

`image_cb` in `tl_detector.py` must be modified as follows. `process_traffic_lights()` must be called only after pose` and `waypoints` are ready (not None).  

```
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

        # This condition is added to deal with some lack due to the 
        # traffic light detection using TensorFlow Object Detection 
        # -------
        # if pose or waypoints isn't ready
        if self.pose is None or self.waypoints is None:
            return  # ignore image_cb

        light_wp, state = self.process_traffic_lights()
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

```

#### Handling Complete Stop

The original implementation does not consider the complete stop in the controller. But now, since we have the red sign from the traffic light, we need to make the car complete stop.

`twist_controller.py` has the control part. I put [400N torque to the brake](https://github.com/arjunbhasin/CarND-Capstone#twist_controllerpy) to make the complete stop. 

```
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_vel = self.vel_lpf.filt(current_vel)

        # rospy.logwarn("Angular vel: {0}".format(angular_vel))
        # rospy.logwarn("Target  vel: {0}".format(linear_vel))
        # rospy.logwarn("Target angular vel: {0}\n".format(angular_vel))
        # rospy.logwarn("Current vel: {0}".format(current_vel))
        # rospy.logwarn("Filtered vel: {0}".format(self.vel_lpf.get()))

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        # original: car slows down but never stop.
        # if linear_vel == 0. and current_vel < 0:
        #     throttle = 0
        #     decel = max(vel_error, self.decel_limit)
        #     brake = abs(decel)*self.vehicle_mass*self.wheel_radius # Torque N*m

        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400     # N to hold the car
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # Torque N*m


        return throttle, brake, steering

```

## Results

### Before TLClassifier implemented

[![output](https://img.youtube.com/vi/GFyBQWzsCyE/0.jpg)](https://youtu.be/GFyBQWzsCyE)

### After TLClassifier implemented - Final Result

[![output](https://img.youtube.com/vi/w3itU_m6iVw/0.jpg)](https://youtu.be/w3itU_m6iVw)


## Troubleshooting

### 404 Error

```
GET /socket.io/?EIO=4&transport=websocket HTTP/1.1" 404 342 0.000446
```

The original requirement of python-socketio is 1.6.1. But I had `404` error from running `styx.launch`.

According to [this](https://github.com/llSourcell/How_to_simulate_a_self_driving_car/issues/34), `python-socketio` must be upgraded inside the current Docker container.

```
pip install python-socketio --upgrade
```

Also, just for others' convenience, I updated `requirements.txt` to have `python-socketio 2.1.2` version

### Camera Error

Inside the simulator, turning on the Camera option will generate an error regarding `cv2_to_imgmsg`. 

According to [this](https://github.com/udacity/CarND-Capstone/issues/107), this problem can be solved by upgrading `pillow` inside the docker. The original version was 2.2.1 and the upgraded one was 5.3.0 at the moment of my upgrade.

```
pip install pillow --upgrade
```


### Waypoint Updater Partial Walkthrough

In the video, two things must be fixed.
- `get_closest_waypoint_id` function name must be `get_closest_waypoint_idx`
- One more condition must be added to the `if` statement inside `loop` function. In my Docker environment, `waypoint_tree` which will be created inside `waypoints_cb` could not ready before `get_closest_waypoint_idx()` was called.

```
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            # original: if self.pose and self.base_waypoints:
            if self.pose and self.base_waypoints and self.waypoint_tree: # add waypoint_tree
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()
```

### Full Waypoint Walkthrough

Without clear notices in the video, some changes were made from the partial walkthrough in this full walkthrough.

```
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            # original: if self.pose and self.base_waypoints:
            ## In FullWaypoint Walkthrough, another change was made here
            #if self.pose and self.base_waypoints and self.waypoint_tree: # add waypoint_tree
            #    closest_waypoint_idx = self.get_closest_waypoint_idx()
            #    self.publish_waypoints(closest_waypoint_idx)
            if self.pose and self.base_line:
                self.publish_waypoints()
            rate.sleep()
```

Another change was made in `base_waypoints` variable of `WaypointUpdater` class. This variable name was replaced with `base_lane` again, without further notices in the full walkthrough video.

```
    def waypoints_cb(self, waypoints):
        # TODO: Implement
        #self.base_waypoints = waypoints
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                    for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

```

### matplotlib 3+

TensorFlow Object Detection API requires matplotlib. If I install matplotlib, its version will be 3+ that requires Python 3.5+. Without it, you won't be able to run the Object Detection APIs. Install an earlier version than 3+.

```
pip install matplotlib==2.1.2
```
### Error in SSD Inception V2 feature extractor
```
raise ValueError('SSD Inception V2 feature extractor always uses' ValueError: SSD Inception V2 feature extractor always usesscope returned by `conv_hyperparams_fn` for both the base feature extractor and the additional layers added since there is no arg_scope defined for the base feature extractor.
```

According to [here](https://github.com/developmentseed/label-maker/commit/94f1863945c47e1b69fe0d6d575caa0b42aa8d63), we can fix it by adding the following line inside the `feature_extractor` section of the config file.

```
      override_base_feature_extractor_hyperparams: true
```


---

## The followings are from the original Udacity's README.md.

---



This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
