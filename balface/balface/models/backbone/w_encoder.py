import math
import torch
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from .helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from .restyle_e4e_encoders import EqualLinear


class WEncoder(Module):
    def __init__(self):
        super(WEncoder, self).__init__()

        num_layers = 50
        mode = 'ir_se'
        output_size = 512

        # print('Using WEncoder')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        log_size = int(math.log(output_size, 2))
        self.style_count = 2 * log_size - 2

        self.load_pretrained()
        for param in self.parameters():
            param.requires_grad = False
    
    def load_pretrained(self):
        ckpt_path = './pretrained/fairface125pad_w_encoder.pt'
        load_state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        encoder_state_dict = dict()
        for key, val in load_state_dict.items():
            if 'encoder' in key:
                encoder_state_dict[key.replace('encoder.', '')] = val
        self.load_state_dict(encoder_state_dict)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x

