import numpy as np
import torch
from PIL import Image
from options.test_options import TestOptions
from models.models import create_model
from runway import RunwayModel
import util.util as util


photosketch = RunwayModel()

@photosketch.setup
def setup():
    global opt
    opt = TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True 
    opt.name = 'pretrained'
    opt.checkpoints_dir = '.'
    opt.model = 'pix2pix'
    opt.which_direction = 'AtoB'
    opt.norm = 'batch'
    opt.input_nc = 3
    opt.output_nc = 1
    opt.which_model_netG = 'resnet_9blocks'
    opt.no_dropout = True
    model = create_model(opt)
    return model


@photosketch.command('convert', inputs={'image': 'image'}, outputs={'converted': 'image'})
def convert(model, inp):
    img = np.array(inp['image'])
    img = img / 255.
    h, w = img.shape[0:2]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float() #.to(device)
    data = {'A_paths': '', 'A': img, 'B': img }
    model.set_input(data)
    model.test()
    output = util.tensor2im(model.fake_B)
    output = Image.fromarray(output.astype('uint8'))
    output = output.convert('RGB')
    return dict(converted=output)


if __name__ == '__main__':
    photosketch.run()
