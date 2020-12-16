from torch.cuda import is_available

def get_device():
    if is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(' - Device: ', device, ' - ')
    return device
