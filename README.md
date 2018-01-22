# Capsule Attention Network

An implementation of a variant of a Capsule Network with attention. 
We use a form of attention to improve signal to noise levels and weight sharing to reduce parameter count.
We force the capsule pose to contain a geometric pose and the attention mechanism uses that information
to guide the 'routing by agreement' algorithm.
The attention mechanism also gives priority to nearby objects as a means to improve the signal to noise ratios for pose estimation.
The attention is and algorithm and is not learned, but the part-to-whole relationships are learned. 
The capsule shares the part-to-whole transformation weights for all children 
instead of having a separate transformation for every child capsule

**Attention algorithm**
- uses geometric pose information to guide the 'routing by agreement' algorithm.
- not learned 
- is not differentiable.  
 
**Weight sharing**
- a Capsule is assigned a fixed number of parts
- part-to-whole transformations weights are shared based on the parent-part-child relationship
- capsules of the same type in the same layer share the part-to-whole weights

**Differences with the paper:**
- Capsule outputs are forced to include geometric pose information that we use for routing.
- Capsule outputs are weighted by the distance between the child and parent capsule
- Part-to-whole transformation weights are 
- We train without the reconstruction regularizaton.

**TODO**
- test with multiple instances per capsule type
- implement rotation and scaling geometric pose
- test with larger images
- more flexible specification of instance and part counts


## Usage

**Step 1.
Install [Keras>=2.0.7](https://github.com/fchollet/keras) 
with [TensorFlow>=1.2](https://github.com/tensorflow/tensorflow) backend.**
```
pip install tensorflow-gpu
pip install keras
```

**Step 2. Clone this repository to local.**
```
git clone https://github.com/eastbayml/CAN.git can
cd can
```

**Step 3. Train a CAN Net on synthetic data**  

First generate some data 
```
python gen_images.py
```
Train with default settings using sample images include in the data/images.npz:

```
python train.py
```

Or use this script to generate new data 
```
python gen_images.py
```

## Results

#### Test Errors   

## Credits

This code borrows heavily from the excellent implementation of the Capsule network by XifengGuo.
 E-mail `guoxifeng1990@163.com`

And of course the paper defining the Capsule concept

[Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)   
