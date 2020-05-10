import os
from time import time
from IPython.display import Audio
from utils import make_batch
from models import Model, Generator
import tensorflow as tf
from scipy.io import wavfile
import numpy as np

tf.compat.v1.disable_eager_execution()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    # inputs, targets = make_batch('voice.wav')
    inputs, targets = make_batch('jaekoo_edit.wav')
    num_time_samples = inputs.shape[1]
    num_channels = 1
    gpu_fraction = 1.0

    print(inputs.shape, targets.shape)

    model = Model(num_time_samples=num_time_samples,
                  num_channels=num_channels,
                  gpu_fraction=gpu_fraction)

    try:
        tic = time()
        model.train(inputs, targets)
    except:
        toc = time()
        print(' +++++ Stopped +++++ ')
    finally:
        print('Training took {} seconds.'.format(toc-tic))
        
        generator = Generator(model)

        # Get first sample of input
        input_ = inputs[:, 0:1, 0]

        tic = time()
        predictions = generator.run(input_, 44100, './')
        toc = time()
        print('Generating took {} seconds.'.format(toc-tic))

        # np.save('pred.npy', predictions)
        wavfile.write('pred.wav', 44100, predictions)
