"""
@author: juschu

saving all things generated during simulation

 - eye position
 - stimulus position
 - output data
 - firing rates and other variables
"""


##############################
#### imports and settings ####
##############################
import os
import numpy as np
import h5py


################
#### saving ####
################
def saveEyePos(eyepos, duration, dirname):
    '''
    save eye position over time as txt-file

    params: eyepos   -- numpy array of eye position over time
                        shape: duration x 2 (eye position in degree at timestep)
            duration -- duration of simulation
            dirname  -- name of folder where to save
    '''

    print('save eye position')

    filename = dirname + 'eyepos.txt'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    strToWrite = ''
    for t in range(duration):
        strToWrite += str(t) + ': ' + str(eyepos[t]) + '\n'
    f = open(filename, 'w')
    f.write(strToWrite)
    f.close()

def saveBarPos(barpos, duration, dirname):
    '''
    save bar positions over time as txt-file

    params: barpos   -- dictionary of bar positions over time
                        key: name of bar, value: numpy array of positions shape (*duration*, 2)
            duration -- duration of simulation
            dirname  -- name of folder where to save
    '''

    print('save bar positions')

    filename = dirname + 'barpos.txt'
    if not os.path.exists(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname))

    strToWrite = ''
    for bar in barpos:
        strToWrite += 'bar: ' + bar + '\n'
        for t in range(duration):
            strToWrite += str(t) + ': ' + str(barpos[bar][t]) + '\n'
        strToWrite += '\n'
    f = open(filename, 'w')
    f.write(strToWrite)
    f.close()

def saveOutput(output, dirname):
    '''
    save output data as txt-file

    params: output  -- dictionary of output data
            dirname -- name of folder where to save
    '''

    print('save output')

    filename = dirname + 'output.txt'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    strToWrite = ''
    for k in output:
        strToWrite += k + ': ' + str(output[k]) + '\n'
    f = open(filename, 'w')
    f.write(strToWrite)
    f.close()

def saveRates(monitors, filename):
    '''
    save recorded rates from monitors as hdf5-file

    params: monitors -- dictionary of ANNarchy's monitor objects
            filename -- name of file where to save
    '''
    
    # recorded_rates = {}
    # for layer in monitors:
    #     recorded_rates[layer] = monitors[layer].get(None, reshape=True)
    # save_dict_to_hdf5(recorded_rates, filename)

    with h5py.File(filename, 'w') as recorded_rates:
        for layer in monitors:
            for key in monitors[layer].variables:
                recorded_rates[f"/{layer}/{key}"] = monitors[layer].get(key, reshape=True)


###########################################
#### store / load dictionary with hdf5 ####
###########################################
def save_dict_to_hdf5(dic, filename):
    """
    see https://codereview.stackexchange.com/questions/120802/
                recursively-save-python-dictionaries-to-hdf5-files-using-h5py

    Save a dictionary whose contents are only strings, np.float64, np.int64,
    np.ndarray, and other dictionaries following this structure
    to an HDF5 file. These are the sorts of dictionaries that are meant
    to be produced by the ReportInterface__to_dict__() method.
    """

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def load_dict_from_hdf5(filename):
    """
    see https://codereview.stackexchange.com/questions/120802/
                recursively-save-python-dictionaries-to-hdf5-files-using-h5py

    Load a dictionary whose contents are only strings, floats, ints,
    numpy arrays, and other dictionaries following this structure
    from an HDF5 file. These dictionaries can then be used to reconstruct
    ReportInterface subclass instances using the
    ReportInterface.__from_dict__() method.
    """

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    see https://codereview.stackexchange.com/questions/120802/
                recursively-save-python-dictionaries-to-hdf5-files-using-h5py

    Take an already open HDF5 file and insert the contents of a dictionary
    at the current path location. Can call itself recursively to fill
    out HDF5 files with the contents of a dictionary.
    """

    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        #print(key,item)
        key = str(key)
        if isinstance(item, list):
            item = np.array(item)
            #print(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, and numpy.float64 types
        # if isinstance(item, (np.int64, np.float64, str, np.float, float, np.float32, int)):
        # `np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself.
        # Doing this will not modify any behavior and is safe.
        if isinstance(item, (np.int64, np.float64, str, float, np.float32, int)):
            #print( 'here' )
            h5file[path + key] = item
            # H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.
            #if not h5file[path + key].value == item:
            if not h5file[path + key][()] == item:
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save numpy arrays
        elif isinstance(item, np.ndarray):
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype('|S9')
                h5file[path + key] = item
            # H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.
            #if not np.array_equal(h5file[path + key].value, item):
            if not np.array_equal(h5file[path + key][()], item):
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        # other types cannot be saved and will result in an error
        else:
            #print(item)
            raise ValueError('Cannot save %s type.' % type(item))

def recursively_load_dict_contents_from_group(h5file, path):
    """
    see https://codereview.stackexchange.com/questions/120802/
                recursively-save-python-dictionaries-to-hdf5-files-using-h5py

    Load contents of an HDF5 group. If further groups are encountered,
    treat them like dicts and continue to load them recursively.
    """

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            # H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.
            #ans[key] = item.value
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans
