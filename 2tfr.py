from random import shuffle
import glob
import os
import sys
import numpy as np
import fs
from fs import open_fs
import tensorflow as tf
import matplotlib.pyplot as plt

# size of object postage stamp cutout in pixels
xdim=32
ydim=32
vdim=5
EXAMPLES=2

tfr_filename = 'TFR.tfr'  # the TFRecord file containing the training set

# Generator function to walk path and generate 1 SP3D image set at a time
def process_example():
  for n in range(EXAMPLES):
    tfrStr = "String %d"%(n)
    tfrInt = np.random.randint(1,100)
    tfrAry = np.random.rand(ydim, xdim)
    tfrAry = tfrAry * 32768
    tfrFlt = np.random.random_sample() * 1000.0
    tfrVec = np.random.rand(vdim)
    yield tfrStr, tfrInt, tfrAry, tfrFlt, tfrVec

  
def _floatvector_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _intvector_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

np.random.seed()

# open the TFRecords file
tfr_writer = tf.python_io.TFRecordWriter(tfr_filename)

# find input files in the target dir "basePath"
# it is critical that pairs are produced reliably first level2 then level1
# for each level2 (Y) file
nExamples = 0
for retStr, retInt, retAry, retFlt, retVec in process_example():
    
  # encode string utf-8 byte vector
  featStr = retStr.encode('utf-8')

  featInt = retInt

  featFlt = retFlt

  featIntVec = np.random.randint(0,10000, size=(vdim))

  featFltVec = retVec

  #visualize retAry as 2D image
  plt.gray()
  img = retAry[0:ydim,0:xdim]
  plt.imshow(img)
  plt.show()
  # flatten array to float vector
  featAry=np.reshape(retAry,(xdim*ydim))
  #featAry=featAry[0:4]

  #featRaw = tf.compat.as_bytes(retAry.tostring())
  #featRaw = retAry.tostring().asbytes()
  featRaw = retAry.tobytes()

  nExamples += 1
  print('\nExample %d'%(nExamples))
  print("%-32.32s"%("String:"), end = ' ')
  print(featStr)
  print("%-32.32s"%("Integer:"), end = ' ')
  print(featInt)
  print("%-32.32s"%("Float:"), end = ' ')
  print(featFlt)
  print("%-32.32s"%("Int Vector:"), end = ' ')
  print(featIntVec)
  print("%-32.32s"%("Float Vector:"), end = ' ')
  print(featFltVec)
  print("%-32.32s"%("Flattened Float 2D Array:"), end = ' ')
  print(featAry[0:4])
  print("%-32.32s"%("Float 2D Array raw byte vector:"), end = ' ')
  print("%40.40s"%(featRaw))

  feature = {
    'string_feature':       _bytes_feature(featStr),
    'integer_feature':      _int_feature(featInt),
    'float_feature':        _float_feature(featFlt),
    'intvector_feature':    _intvector_feature(featIntVec),
    'floatvector_feature':  _floatvector_feature(featFltVec.tolist()),
    'array_feature':        _floatvector_feature(featAry.tolist()),
    'raw_feature':          _bytes_feature(featRaw),
  }

  # Create an example protocol buffer
  example = tf.train.Example(features=tf.train.Features(feature=feature))

  # write the example to the TFRecordWriter
  # Serialize to string and write on the file
  tfr_writer.write(example.SerializeToString())

  sys.stdout.flush()
  
tfr_writer.close()
print('%d examples'%(nExamples))
sys.stdout.flush()

