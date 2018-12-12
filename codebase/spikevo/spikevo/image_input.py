import numpy as np

RATE = 'rate'
ON_OFF = 'on-off'
IMAGE_ENCODINGS = [RATE, ON_OFF]
class SpikeImage(object):
    
    def __init__(self, width, height, encoding=ON_OFF, encoding_params={}):
        if encoding not in IMAGE_ENCODINGS:
            raise Exception("Image encoding not supported ({})".format(encoding))

        self._width = width
        self._height = height
        self._size = width * height
        self._encoding = encoding
        self._enc_params = encoding_params
        
        
    def _encode_on_off(self, source):
        flat = source.flatten()
        rate = self._enc_params['rate']
        thresh = self._enc_params['threshold']
        off_pixels = np.where(flat < threshold)[0]
        on_pixels = np.where(flat >= threshold)[0]
        
        return [rate*flat[i] if i in on_pixels else 0 for i in range(self._size)],
            [rate*flat[i] if i in off_pixels else 0 for i in range(self._size)]
    
    def _encode_rate(self, source):
        flat = source.flatten()
        rate = self._enc_params['rate']
        return [rate*flat[i] for i in range(self._size)]
        
    def encode(self, source):
        if self._encoding == RATE:
            return self._encode_rate(source)
        elif self._encoding == ON_OFF:
            return self._encode_rate(source)
        
