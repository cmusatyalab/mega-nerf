import re
import numpy as np
import sys

'''
Read a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def read_pfm(file):
  file = open(file, 'rb')
  
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().rstrip()
  if header.decode('ascii') == 'PF':
    color = True    
  elif header.decode('ascii') == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.search(r'(\d+)\s(\d+)', file.readline().decode('ascii'))
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.reshape(data, shape).astype(np.float32), scale

'''
Write a Numpy array to a PFM file.
'''
def write_pfm(file, image, scale = 1):
  file = open(file, 'wb')

  color = None


  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write(b'PF\n' if color else b'Pf\n')
  file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write(b'%f\n' % scale)

  image.tofile(file)  