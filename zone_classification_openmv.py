# TensorFlow Lite Mobilenet V1 Example
#
# Google's Mobilenet V1 detects 1000 classes of objects
#
# WARNING: Mobilenet is trained on ImageNet and isn't meant to classify anything
# in the real world. It's just designed to score well on the ImageNet dataset.
# This example just shows off running mobilenet on the OpenMV Cam. However, the
# default model is not really usable for anything. You have to use transfer
# learning to apply the model to a target problem by re-training the model.
#
# NOTE: This example only works on the OpenMV Cam H7 Pro (that has SDRAM) and better!
# To get the models please see the CNN Network library in OpenMV IDE under
# Tools -> Machine Vision. The labels are there too.
# You should insert a microSD card into your camera and copy-paste the mobilenet_labels.txt
# file and your chosen model into the root folder for ths script to work.
#
# In this example we slide the detector window over the image and get a list
# of activations. Note that use a CNN with a sliding window is extremely compute
# expensive so for an exhaustive search do not expect the CNN to be real-time.

import sensor, image, time, os, tf, utime

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.GRAYSCALE)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.HQQVGA)      # Set frame size to QVGA (320x240)
#sensor.set_windowing((48, 48))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.
"""
mobilenet_version = "1" # 1
mobilenet_width = "0.5" # 1.0, 0.75, 0.50, 0.25
mobilenet_resolution = "128" # 224, 192, 160, 128
"""
mobilenet = "NEW_model3_24E_92_quantized3.tflite"
labels = [line.rstrip('\r\n') for line in open("classes.txt")]

datasetfolder = 'rsd_dts/'

def evaluate_image(im,tag):
     img = image.Image(im, copy_to_fb = True)
     start = utime.ticks_ms() # get value of millisecond counter

     for obj in tf.classify(mobilenet, img):
        print(im)
        print(tag)
        probabilities = obj[4]
        print(probabilities)
        sorted_list = sorted(zip(labels, obj.output()), key = lambda x: x[1], reverse = True)
        print(sorted_list)
     delta = utime.ticks_diff(utime.ticks_ms(), start) # compute time difference
     print(str(delta)+"ms")

def evaluate_frame(im):
     start = utime.ticks_ms() # get value of millisecond counter

     for obj in tf.classify(mobilenet, img):
        probabilities = obj[4]
        print(probabilities)
        sorted_list = sorted(zip(labels, obj.output()), key = lambda x: x[1], reverse = True)
        print(sorted_list)
     delta = utime.ticks_diff(utime.ticks_ms(), start) # compute time difference
     print(str(delta)+"ms")


clock = time.clock()

enable_dataset = False
if(enable_dataset):
    for tag in labels:
        data_path = datasetfolder + tag
        for file in os.listdir(data_path):
            evaluate_image(data_path + '/' + file, tag)



while(True):
    clock.tick()

    img = sensor.snapshot()
    evaluate_frame(img)
    print(clock.fps(), "fps")

