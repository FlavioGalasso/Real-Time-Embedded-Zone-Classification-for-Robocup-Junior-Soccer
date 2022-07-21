
import sensor, image, time, os, tf, utime

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.GRAYSCALE)    # Set pixel format to GRAYSCALE
sensor.set_framesize(sensor.HQQVGA)      # Set frame size to HQQVGA
#sensor.set_windowing((48, 48))       # Set 48x48 window but actually the model resizes the image by itself in tf.classify function

sensor.skip_frames(time=2000)          # Let the camera adjust.


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

enable_dataset = False # if this is true it loads the validation dataset from the sdcard and prints out the predictions, framerate, inference time for each image
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

