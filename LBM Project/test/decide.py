from numba import cuda


def run(target_function, target_device, *args, **kwargs):
    new_func_name = target_function + '_' + target_device
    if target_device == 'cpu':
        try:
            eval(new_func_name)(**kwargs)
            print('run %s on %s successfully' % (target_function, target_device))
        except:
            print('failed to run %s on %s' % (str(target_function), str(target_device)))
            return 0
        else:
            return 1
    elif target_device == 'gpu':
        for key in kwargs:
            if key == 'TPB':
                TPB = kwargs[key]
            elif key == 'BPG':
                BPG = kwargs[key]
            else:
                print('key %s cannot be recognized' % key)
        print('TPB,BPG:', TPB, BPG)
        try:
            eval(new_func_name)[TPB, BPG](*args)
            print('run %s on %s successfully' % (target_function, target_device))
        except:
            print('failed to run %s on %s' % (str(target_function), str(target_device)))
            return 0
        else:
            return 1
    else:
        print('please input correct target device type (cpu / gpu)')
        return 0


def gpu_parameter_loading_from_cpu(**kwargs):
    """if parameter exist on CPU then copy to GPU"""
    for key in kwargs:
        try:
            if 'float' in str(type(kwargs[key])) or 'int' in str(type(kwargs[key])):
                """float and int type cannot load to GPU, but they could be directly used by GPU"""
                globals()[str(key) + '_global'] = kwargs[key]
            else:
                """create a new parameter name with _global and assign the cpu """
                globals()[str(key) + '_global'] = cuda.to_device(kwargs[key])
            print('%s load to GPU successfully' % key)
        except:
            print('%s has not been initialize in CPU' % key)
            return 0

    return 1


def cpu_parameter_loading_from_gpu(**kwargs):
    """if parameter exist on GPU then retrieve"""
    for key in kwargs:
        func_name_on_gpu = eval(str(key) + '_global')
        """tell if exist on GPU"""
        try:
            parameter_type = type(func_name_on_gpu)
            """if on GPU"""
            if 'cuda' in str(parameter_type):
                kwargs[key] = func_name_on_gpu.copy_to_host
                """if not on GPU"""
                print('%s load to cpu successfully' % key)
            else:
                print('%s is not on GPU, the original cpu version of %s is used' % (key, key))
        except:
            print('the parameter %s is not exist on both CPU and GPU' % key)
            return 0
    return 1
