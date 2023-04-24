import time
import numpy as np

import minimalmodbus
import serial


class LIR2:
    def read_firmware_version(self) -> str:
        '''Read the firmware version of the sensor

        Parameters:
        Returns:
            firmware_version (str): Firmware version
        '''
        raw = str(self.sensor.read_register(0, 0, 4))
        return f"Firmware version: {raw[:2]}.{raw[2:]}"

    def read_sample(self, sample:int) -> int:
        '''Read the specified sample from the sensor

        Parameters:
            sample (int): index of the pixel from which the sample should be taken (starts with 0)
        Returns:
            sample (float): Temperature sample from the specified pixel in °C
        '''
        try:
            return self.sensor.read_register(9 + sample, 0, 4) / 100
        except IOError:
            return -1

    def read_samples(self) -> np.ndarray:
        '''Read samples from all 192 pixels of the sensor

        Parameters:
        Returns:
            samples (numpy.ndarray): A 16 x 12 array of float values °C
        '''
        try:
            raw_data = [ val for reg in range(9, 200, 96) for val in self.sensor.read_registers(reg, 96, 4) ]
            self.matrix = np.array(np.split(np.array(raw_data), 12)) / 100
            return self.matrix
        except IOError as e:
            print(e)
            return self.matrix

    def read_raw_light_intensity(self) -> int:
        '''Read raw light intensity from the sensor

        Parameters:
        Returns:
            intensity (int): raw light intensity from the sensor
        '''
        return self.sensor.read_register(201, 0, 4)

    def read_light_intensity(self) -> int:
        '''Read light intensity from the sensor

        Parameters:
        Returns:
            intensity (int): scaled light intensity (raw / 100)
        '''
        return self.sensor.read_register(202, 0, 4) / 100

    def read_filtered_light_intensity(self) -> int:
        '''Read filtered light intensity

        Parameters:
        Returns:
            intensity (int): scaled filtered light intensity (raw / 100)
        '''
        return self.sensor.read_register(203, 0, 4) / 100

    def read_result_area(self, area:int) -> int:
        '''Reads the specified result area

        Parameters:
            area (int): index of the area (starts with 0)
        Returns:
            res_area (int): scaled result area (raw / 100)
        '''
        return self.sensor.read_register(204 + area, 0, 4) / 100

    def read_result_areas(self) -> list:
        '''Reads result areas
        
        Paramaeters:
        Returns:
            res_areas (list): a list of scaled result areas (raw / 100)
        '''
        return [t / 100 for t in self.sensor.read_registers(204, 5, 4)]
    
    def __test_connection(self):
        try:
            self.sensor.read_register(0,0,4)
            return True
        except IOError:
            return False
        


    def __init__(self, port, slave_address) -> None:
        self.sensor = minimalmodbus.Instrument(port, slave_address, minimalmodbus.MODE_RTU, True, False)
        self.matrix = np.zeros((12, 16))

        # Init serial connection
        self.sensor.serial.port = port
        self.sensor.serial.baudrate = 115200
        self.sensor.serial.parity = serial.PARITY_NONE
        self.sensor.serial.stopbits = 1
        self.sensor.serial.bytesize = 8
        self.sensor.serial.timeout = 1
        self.sensor.serial.write_timeout = 1

        for _ in range(5):
            if self.__test_connection():
                break
            time.sleep(0.2)
        else:
            raise IOError('Connection failed after 5 attempts')
                
