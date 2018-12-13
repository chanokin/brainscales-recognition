import numpy as np

RED, GREEN, BLUE = range(3)
ON, OFF = range(2)
CHAN2COLOR = {ON: GREEN, OFF: RED}
CHAN2TXT = {ON: 'GREEN', OFF: 'RED'}
RATE = 'rate'
ON_OFF = 'on-off'
IMAGE_ENCODINGS = [RATE, ON_OFF]
class SpikeImage(object):
    
    def __init__(self, width, height, encoding=ON_OFF, 
        encoding_params={'rate': 100, 'threshold': 12}):
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
        threshold = self._enc_params['threshold']
        off_pixels = np.where(flat < threshold)[0]
        on_pixels = np.where(flat > threshold)[0]
        
        return {ON: [rate*(flat[i] - threshold) if i in on_pixels else 0 \
                                            for i in range(self._size)],
                OFF: [rate*(threshold - flat[i]) if i in off_pixels else 0 \
                                            for i in range(self._size)]}


    def _encode_rate(self, source):
        flat = source.flatten()
        rate = self._enc_params['rate']
        return {OFF: [rate*flat[i] for i in range(self._size)]}


    def encode(self, source):
        if self._encoding == RATE:
            return self._encode_rate(source)
        elif self._encoding == ON_OFF:
            return self._encode_on_off(source)


    def rate_to_poisson(self, spike_rates, start_time, end_time):
        def nextTime(rateParameter):
            return -np.log(1.0 - np.random.random()) / rateParameter

        def poisson_generator(rate, t_start, t_stop):
            poisson_train = []
            if rate > 0:
                next_isi = nextTime(rate)*1000.
                last_time = next_isi + t_start
                while last_time  < t_stop:
                    poisson_train.append(last_time)
                    next_isi = nextTime(rate)*1000.
                    last_time += next_isi
            return poisson_train
        
        return [poisson_generator(rate, start_time, end_time) \
                                        for rate in spike_rates]


    def spikes_to_image(self, spike_trains, tstart=0, tend=np.inf, 
        spike_val=1):
        """spike_trains is a dictionary containing spikes from each channel
            (2 for ON_OFF encoding, 1 for RATE encoding)
            
            Spike times have to be in a format compatible with the PyNN 0.8+
            [[n0t0, n0t1], [n1t0, n1t1, n1t2] ... [nNt0]]
        """
        channels = 3 if self._encoding == ON_OFF else 1
        img = np.zeros((self._height, self._width, channels))
        for ch in spike_trains:
            color = CHAN2COLOR[ch]
            for nid, spike_times in enumerate(spike_trains[ch]):
                if len(spike_times) == 0:
                    continue
                row, col = nid//self._width, nid%self._width
                # print(CHAN2TXT[ch], row, col, spike_times)
                print(CHAN2TXT[ch], row, col)
                for t in spike_times:
                    img[row, col, color] += spike_val
        
        return img

    def create_pops(self, pynnx, neuron_parameters={ON:{}, OFF:{}},
        generation_type='array'):
        if generation_type == 'array':
            return self.create_pops_array(pynnx, neuron_parameters=neuron_parameters)
        elif generation_type == 'poisson':
            return self.create_pops_poisson(pynnx, neuron_parameters=neuron_parameters)

    def create_pops_poisson(self, pynnx, neuron_parameters={ON:{}, OFF:{}}):
        if self._encoding == RATE:
            """OFF means first and only channel for RATE encoding,
                this gets translated into a RED == 0 index in the image
            """
            return {
                OFF: pynnx.Pop(self._size, pynnx.sim.SpikeSourcePoisson, 
                    neuron_parameters, label='Rate encoded image')
            }
        elif self._encoding == ON_OFF:
            return {
                OFF: pynnx.Pop(self._size, pynnx.sim.SpikeSourcePoisson, 
                    neuron_parameters[OFF], label='OFF - rate encoded image'),
                ON: pynnx.Pop(self._size, pynnx.sim.SpikeSourcePoisson, 
                    neuron_parameters[ON], label='ON - rate encoded image')
            }

    def create_pops_array(self, pynnx, neuron_parameters={ON:{}, OFF:{}}):
        if self._encoding == RATE:
            """OFF means first and only channel for RATE encoding,
                this gets translated into a RED == 0 index in the image
            """
            return {
                OFF: pynnx.Pop(self._size, pynnx.sim.SpikeSourceArray, 
                    neuron_parameters, label='Rate encoded image')
            }
        elif self._encoding == ON_OFF:
            return {
                OFF: pynnx.Pop(self._size, pynnx.sim.SpikeSourceArray, 
                    neuron_parameters[OFF], label='OFF - rate encoded image'),
                ON: pynnx.Pop(self._size, pynnx.sim.SpikeSourceArray, 
                    neuron_parameters[ON], label='ON - rate encoded image')
            }

