
   
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
    
import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops 

import torch    
import torch.nn as nn     
from engine.extre_module.ultralytics_nn.conv import Conv, RepConv, autopad
from engine.extre_module.torch_utils import model_fuse_test


class RGELAN(nn.Module):
  
     
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):  
        super(RGELAN, self).__init__()    
        self.c = int(c2 * e)  
        self.mid = int(self.c * scale)    
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        self.cv3 = RepConv(self.c, self.mid, 3) 
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))   
        self.cv4 = Conv(self.mid, self.mid, 1)
        
    def forward(self, x):
        
        y = list(self.cv1(x).chunk(2, 1))   
        y[-1] = self.cv3(y[-1]) 
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))  
         
        return self.cv2(torch.cat(y, 1))   
    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = RGELAN(in_channel, out_channel, n=2, scale=0.5, e=0.5).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(GREEN + 'test reparameterization.' + RESET)
    module = model_fuse_test(module)
    outputs = module(inputs) 
    print(GREEN + 'test reparameterization done.' + RESET)
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True,   
                                     output_precision=4,   
                                     print_detailed=True)  
    print(RESET)    
