import tensorflow as tf

import logging

def set_logger():
    logger = logging.getLogger('main')
    logger.setLevel(logger.)

def check_device():
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus):
        return gpus[0].name
    else:
        cpu = tf.config.list_physical_devices('CPU')[0]
        return cpu.name

if __name__ == '__main__':
    device = check_device()
