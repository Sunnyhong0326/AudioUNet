import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    
    # path of experiment results
    parser.add_argument('--result_path', type = str, default = 'results', help = 'the result path')
    parser.add_argument('--exp' ,type = str, required = True, help = 'experiment name')

    # hyperparameter info
    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
    parser.add_argument('--num_epochs', type = int, default = 5000, help = 'number of epochs')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate')
    parser.add_argument('--act', type = str, default = 'relu', help = 'activate function',\
                        choices = ['relu','sine'])
    
    # model info
    parser.add_argument('--num_blocks', type = int, default = 4, help = 'number of downsample or upsample blocks')
    parser.add_argument('--max_channel', type = int, default = 512, help = 'maximum channels of each layer output')
    parser.add_argument('--num_filter_sizes', type = list, default = [65, 33, 17,  9,  9,  9,  9, 9, 9], \
                        help = 'filter size for each layer')
    parser.add_argument('--num_filters', type = list, default = [128, 256, 512, 512, 512, 512, 512, 512], \
                        help = 'number of filters for each layer')
    parser.add_argument('--patch_size', type = int, default = 8192, help = "input length that can feed into model")


    # dataset info
    parser.add_argument('--audio_path', type = str, default = './data', help = 'dataset path')
    parser.add_argument('--in_sr', type = int, default = 12000, help = "input sample rate")
    parser.add_argument('--out_sr', type = int, default = 48000, help = "output sample rate")
    parser.add_argument('--up_scale',  type = int, default = 4, help = "up scaling factor")

    opts = parser.parse_args()
    return opts   
    
if __name__== '__main__':
    opts = get_opts()
    print(opts)