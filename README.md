# A Complete Example of Tensorflow Queues 

The story behind this is that I kept seeing mention of Tensorflow Queues and that `feed_dict` shouldn't be used in serious Tensorflow code, but the [documentation](https://www.tensorflow.org/programmers_guide/reading_data) didn't have a complete example of how to use them. [This](http://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/) post
was the most helpful resource I found when I was learning how to use Queues. 

I have provided a <b>complete</b> example (albeit simple) that covers all the way from cleaning the data to evaluating the model on a test set, all the while using Tensorflow Queues.
It's pretty neat when you realize you don't have to use `feed_dict` because the underlying Session graph already knows where to get your train/validation/test data! 

On my NVIDIA GTX 1080, this takes up about 6 GB since the data can be transferred directly onto the GPU. The model itself is only a few hundred MBs. This is much faster than having to copy each mini-batch from the CPU -> GPU each time you call `feed_dict`; in fact, the data is copied over [twice](https://groups.google.com/a/tensorflow.org/d/msg/discuss/SXWDjrz5kZw/Oj1PO_RnBQAJ).

This example is a modified version of the Convolutional Neural Network from this [tutorial](https://www.tensorflow.org/get_started/mnist/pros).

The MNIST dataset is included under `MNIST_data`. Run
```python
python mnist-to-jpg.py
```
to unpack the data, then do

```python
python cnn.py
```
It should take about half an hour to run on a CPU, or a few minutes on a GPU.

The code has been tested with Python 3.5 and Tensorflow 1.1. 

If using Anaconda, make sure to have `scipy` and `Pillow` also installed. 
