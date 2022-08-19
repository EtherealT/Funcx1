from numba import cuda


def gpu_parameter_loading_from_cpu(**kwargs):
    """if parameter exist on CPU then copy to GPU"""
    for key in kwargs:
        try:
            """create a new parameter name with _global and assign the cpu """
            locals()[str(key) + '_global'] = cuda.to_device(key)
        except:
            print('%s has not been initialize in CPU' % key)
            return 0
        else:
            return 1


def cpu_parameter_loading_from_gpu(**kwargs):
    """if parameter exist on GPU then retrieve"""
    for key in kwargs:
        func_name_on_gpu = str(key) + '_global'
        """tell if exist on GPU"""
        parameter_type = type(func_name_on_gpu)
        """if on GPU"""
        if 'cuda' in str(parameter_type):
            kwargs[key] = func_name_on_gpu.copy_to_host
        """if not on GPU"""
    else:
        print('%s is not on GPU, the original cpu version of %s is used' % (key, key))
        return 0
    return 1


def run(target_function, target_device, TPB=1, BPG=1, **kwargs):
    new_func_name = target_function.__name__ + '_' + target_device
    if target_device == 'cpu':
        try:
            new_func_name(**kwargs)
        except:
            print('failed to run %s on %s' % (str(target_function), str(target_device)))
            return 0
        else:
            return 1
    elif target_device == 'gpu':
        try:
            new_func_name[TPB, BPG](**kwargs)
        except:
            print('failed to run %s on %s' % (str(target_function), str(target_device)))
            return 0
        else:
            return 1
    else:
        print('please input correct target device type (cpu / gpu)')
        return 0

