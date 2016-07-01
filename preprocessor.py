import numpy as np 
import h5py
import time 

class TextProcessor(object):

    def __init__(self,source_file):
        self.source_file = source_file

    def grab_dat(self):
        """
        Grab data from the source file and turn it into a string of indices that correspond
        to characters in a master list. 
        
        """
        dat_str = str()
        with open(self.source_file,'r') as reader:
            for line in reader:
                dat_str += line 
        dat_str = dat_str.lower() 
        unique_char = list(set(dat_str))
        unique_char.sort()  
        char_dict = {char:i for i, char in enumerate(unique_char)}
        dat_small = np.zeros((len(dat_str)),dtype=int)
        for i,char in enumerate(dat_str):
            dat_small[i] = char_dict[char]
        
        return dat_small, char_dict, unique_char

    def save_dat(self,filename):

        dat_small, char_dict, unique_char = self.grab_dat() 
        t0 = time.time()
        print("Saving dataset") 
        f= h5py.File(filename, 'w') 
        f.create_dataset('dat',data=dat_small)
        f.attrs['chars'] = unique_char 
        f.close() 
        print("Took {:.2f} seconds to save dataset".format(time.time()-t0))
        

if __name__ == '__main__':

    txter = TextProcessor('shakespeare.txt')
    txter.save_dat("shakespeare.hdf5")  





