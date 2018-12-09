import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

XDim=32
YDim=32
ZDim=1
VDim=5

pathTFR = 'TFR.tfr'  # The TFRecord file containing the data

batchSize=1
batchN=2

with tf.Session() as sess:
    feature = {
        'string_feature':        tf.FixedLenFeature(shape=[], dtype=tf.string),
        'integer_feature':       tf.FixedLenFeature(shape=[], dtype=tf.int64),
        'float_feature':         tf.FixedLenFeature(shape=[], dtype=tf.float32),
        'intvector_feature':     tf.FixedLenFeature(shape=[VDim], dtype=tf.int64),
        'floatvector_feature':   tf.FixedLenFeature(shape=[VDim], dtype=tf.float32),
        'array_feature':         tf.FixedLenFeature(shape=[YDim*XDim], dtype=tf.float32),
        'raw_feature':           tf.FixedLenFeature(shape=[], dtype=tf.string),
    }
        #'string_feature':        tf.VarLenFeature(dtype=tf.string),
        #'integer_feature':       tf.VarLenFeature(dtype=tf.int64),
        #'params': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True)
        
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([pathTFR], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    """Parse a single record into x and y images"""
    featStr = features['string_feature']
    featInt = features['integer_feature']
    featFlt = features['float_feature']
    featIntVec = features['intvector_feature']
    featFltVec = features['floatvector_feature']
    featAry = features['array_feature']
    featRaw = features['raw_feature']

    #x = tf.slice(x, (0, 0, 0, 0), (WStokes, YDim, XDim, ZStokes))
    
    # Any preprocessing here ...
    # Convert the image data from string back to the numbers
    #featImage = tf.decode_raw(featRaw, tf.float32)
    # Reshape image data into the original shape
    #featImage = tf.reshape(featImage, [YDim, XDim])
    
    # Cast label data into int32
    #label = tf.cast(features['train/label'], tf.int32)

    # Creates batches by randomly shuffling tensors
    tenStr, tenInt, tenFlt, tenIntVec, tenFltVec, tenAry, tenRaw = tf.train.batch([featStr, featInt, featFlt, featIntVec, featFltVec, featAry, featRaw], batch_size=batchSize)

    tenStr = tenStr[0]
    tenInt = tenInt[0]
    tenFlt = tenFlt[0]
    tenIntVec = tenIntVec[0]
    tenFltVec = tenFltVec[0]
    tenAry = tenAry[0]
    tenAry = tf.reshape(tenAry, (YDim, XDim))
    tenRaw = tenRaw[0]
    # Convert the image data from string back to the numbers
    tenImg = tf.decode_raw(tenRaw, tf.float64)
    # Reshape image data into the original shape
    tenImg = tf.reshape(tenImg, [YDim, XDim])
    #tenImage = tenImage[0]

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    plt.gray()
    # Now we read batches of images and labels and plot them
    for batch_index in range(batchN):
        valStr, valInt, valFlt, valIntVec, valFltVec, valAry, valRaw, valImg = sess.run([tenStr, tenInt, tenFlt, tenIntVec, tenFltVec, tenAry, tenRaw, tenImg ])
        for i in range(batchSize):
            #print(batStr[0].values[i])
            #print(batInt[i])
            print('\nExample %d'%(i+1))
            print("%-32.32s"%("String:"), end = ' ')
            print(valStr)
            print("%-32.32s"%("Integer:"), end = ' ')
            print(valInt)
            print("%-32.32s"%("Float:"), end = ' ')
            print(valFlt)
            print("%-32.32s"%("Int Vector:"), end = ' ')
            print(valIntVec)
            print("%-32.32s"%("Float Vector:"), end = ' ')
            print(valFltVec)
            print("%-32.32s"%("2D Array:"), end = ' ')
            print(valAry[0][0:4])
            print("%-32.32s"%("Float 2D Array raw byte vector:"), end = ' ')
            print("%40.40s"%(valRaw))
            img = valAry[0:YDim,0:XDim]
            plt.imshow(img)
            plt.show()
            img = valImg[0:YDim,0:XDim]
            plt.imshow(img)
            plt.show()


    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()

