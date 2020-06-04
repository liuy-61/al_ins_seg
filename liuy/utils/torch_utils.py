import torch
import os
import pickle

from liuy.utils.local_cofig import OUTPUT_DIR
# from liuy.implementation.InsSegModel import OUTPUT_DIR
def select_device(device='', apex=False):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        cuda_str = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                cuda_str = ' ' * len(cuda_str)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (cuda_str, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

def load_prj_model(project_id):
    """Read existed project model or init a new model."""
    device = select_device()
    detail_output_dir = os.path.join(OUTPUT_DIR, 'project_' + project_id)
    if os.path.exists(os.path.join(detail_output_dir, project_id + '_model.pkl')):
        with open(os.path.join(detail_output_dir, project_id + '_model.pkl'), 'rb') as f:
            model_ft = pickle.load(f)
    else:
        return None, device

    model_ft = model_ft.to(device)
    return model_ft, device

def load_regression_model(regression_id):
    device = select_device()
    detail_output_dir = os.path.join(OUTPUT_DIR, 'project_' + regression_id)


