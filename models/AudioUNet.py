import torch
import torch.nn as nn
from .PixelShuffle1D import PixelShuffle1D

class AudioUNet(nn.Module):
    """
    self.blocks: number of blocks in downsample/upsample process
    Note: One block consists of multiple layers
    self.n_filters: number of filters for each downsample/upsample block
    self.n_filter_sizes: filter length for each downsample/upsample block
    self.D_blocks: all downsample blocks
    self.B_block: bottleneck block
    self.U_blocks: all upsample blocks
    self.F_block: final block
    """
    def __init__(self,
                 n_filters,
                 n_filter_sizes,
                 up_scale = 4,
                 num_blocks = 4,
                 debug = False):
        """
        num_layers: number of blocks in downsample/upsample block
        n_filters: number of filters for each block
        n_filters_sizes: filter length for each block
        up_scale: up scaling factor
        """
        super(AudioUNet, self).__init__()

        self.blocks = num_blocks

        # filter info
        self.n_filters = n_filters # default: [65, 33, 17,  9,  9,  9,  9, 9, 9]
        self.n_filter_sizes = n_filter_sizes # default [128, 256, 512, 512, 512, 512, 512, 512]

        # in out channel info
        in_channel = 1
        out_channel = None

        # up scale
        self.up_scale = up_scale*2

        # downsample blocks 
        self.D_blocks = nn.ModuleList()
        for n_blk in range(self.blocks):
            out_channel = self.n_filters[n_blk]
            if debug:
                print('in_channel:', in_channel, 'out_channel:', out_channel)
            D_block = self.downsample_block(in_channel, out_channel, self.n_filter_sizes[n_blk])
            self.D_blocks.append(D_block)
            in_channel = out_channel

        # bottleneck blocks
        out_channel = self.n_filters[-1]
        if debug:
            print('BottleNeck','in_channel:', in_channel, 'out_channel:', out_channel)
        self.B_block = self.bottleneck_block(in_channel, out_channel, self.n_filter_sizes[-1])

        # upsampling blocks
        self.U_blocks = nn.ModuleList()
        in_channel = out_channel
        for n_blk in reversed(range(self.blocks)):
            out_channel = self.n_filters[n_blk]*2
            if debug:
                print('in_channel:', in_channel, 'out_channel:', out_channel)
            U_block = self.upsample_block(in_channel, out_channel, self.n_filter_sizes[n_blk])
            self.U_blocks.append(U_block)
            in_channel = out_channel

        # final conv blocks
        self.F_block = self.final_conv(in_channel, self.up_scale, 9, self.up_scale)

    def downsample_block(self, in_channel, out_channel, filter_size):
        block = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channel, 
                out_channels = out_channel, 
                kernel_size = filter_size,
                padding = 'same'),
            nn.MaxPool1d(
                kernel_size = 2
            ),
            nn.LeakyReLU(
                negative_slope = 0.2,
                inplace = True
            )
        )
        return block
    
    def upsample_block(self, in_channel, out_channel, filter_size):
        block = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channel, 
                out_channels = out_channel, 
                kernel_size = filter_size,
                padding = 'same'),
            nn.Dropout(p = 0.5),
            nn.ReLU(inplace = True),
            PixelShuffle1D(2)
        )
        return block
    
    def bottleneck_block(self, in_channel, out_channel, filter_size):
        block = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channel,
                out_channels = out_channel,
                kernel_size = filter_size,
                padding = 'same'
            ),
            nn.MaxPool1d(
                kernel_size = 2
            ),
            nn.Dropout(p = 0.5),
            nn.LeakyReLU(
                negative_slope = 0.2,
                inplace = True
            )
        )
        return block
    

    def final_conv(self, in_channel, out_channel, filter_size, up):
        block = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channel,
                out_channels = out_channel,
                kernel_size = filter_size,
                padding = 'same'
            ),
            PixelShuffle1D(up)
        )
        return block

    def forward(self, x):
        # save input for final skip summation
        input_x = x

        # block number for debug use
        blk_num = 0

        # list for skip connections
        down_sample_l = list()

        #print('DownSample:')
        # downsample propagation
        for n_blk in range(self.blocks):
            #print(' Block:', blk_num)
            x = self.D_blocks[n_blk](x)
            down_sample_l.append(x)
            #print(' Shape: ', x.shape)
            blk_num += 1

        # bottleneck propagation
        #print('Bottleneck:')
        #print(' Block:', blk_num)
        x = self.B_block(x)
        #print(' Shape: ', x.shape)
        blk_num += 1

        # upsample propagation
        #print('Upsample:')
        for n_blk in range(self.blocks):
            #print(' Block:', blk_num)
            # skip connection from downsample features
            x = self.U_blocks[n_blk](x)
            #print(' x shape: ', x.shape)
            #print(' Downsample', down_sample_l[self.blocks - n_blk - 1].shape)
            x = torch.cat([x, down_sample_l[self.blocks - n_blk - 1]], dim = 1)
            #print(' Concat Shape: ', x.shape)
            
            #print(' output shape', x.shape)
            blk_num += 1
        
        # final conv propagation
        x = self.F_block(x)
        #print('Shape: ', x.shape)

        return x


