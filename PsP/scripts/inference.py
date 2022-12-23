import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
import os
import pandas as pd
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def getTransforms(mode):
    transformList = []
    newSize = (256, 256)

    if mode == 'img':
        transformList.append(transforms.Resize(newSize, interpolation=InterpolationMode.BICUBIC))
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif mode == 'anno':
        transformList.append(transforms.Resize(newSize, interpolation=InterpolationMode.NEAREST))
        transformList.append(transforms.ToTensor())

    return transforms.Compose(transformList)


def RGBAnno2Mask(anno, mappingDict:dict):
    seg = np.array(anno)

    R = seg[:,:,0]
    G = seg[:,:,1]
    B = seg[:,:,2]
    ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32))
    uniqueClasses = np.unique(ObjectClassMasks)

    maxIdx = max(mappingDict.values())
    for c in uniqueClasses:
        if c not in mappingDict:
            maxIdx+=1
            mappingDict[c] = maxIdx

    func = np.vectorize(lambda x, *y:mappingDict[x])
    ObjectClassMasks = func(ObjectClassMasks)

    return Image.fromarray(np.uint8(ObjectClassMasks)), mappingDict


def convertAnnoTensor(annoTensor:torch.Tensor, styleSize:int) -> torch.Tensor:
    """
    convert annoTensor from label encoding to one-hot encoding

    Args:
        annoTensor: segmentation map
        styleSize: number of classes
    
    Returns:
        oneHotEncondingTensor
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    annoTensor = annoTensor.long()
    batchSize, _, h, w = annoTensor.size()
    oneHotEncondingTensor = torch.FloatTensor(batchSize, styleSize, h, w).zero_().to(device)
    oneHotEncondingTensor.scatter_(1, annoTensor, 1.0)
        
    return oneHotEncondingTensor

def run():
    # test_opts is a dictionary
    test_opts = TestOptions().parse()
    print(test_opts)


    # initial some parameters
    test_opts.checkpoint_path="../PsP_ckpt/iteration_200000.pt"
    # print(os.getcwd())
    # print(test_opts.checkpoint_path)
    test_opts.test_batch_size=1
    test_opts.test_workers=1

    
    # if test_opts.resize_factors is not None:
    #     assert len(
    #         test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
    #     out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
    #                                     'downsampling_{}'.format(test_opts.resize_factors))
    #     out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
    #                                     'downsampling_{}'.format(test_opts.resize_factors))
    # else:
    #     out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    #     out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    # os.makedirs(out_path_results, exist_ok=True)
    # os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    df = pd.read_csv('../SPADE/mappingfiles/Name2Idx.csv', encoding="UTF-8")
    mappingDict = dict(df.loc[:, 'OriginalIdx':'NewIdx'].to_dict('split')['data'])



    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    # print('Loading dataset for {}'.format(opts.dataset_type))


    # dataset_args = data_configs.DATASETS[opts.dataset_type]
    # transforms_dict = dataset_args['transforms'](opts).get_transforms()
    # transform = transforms_dict['transform_inference']
    one_image = '../testfolder/SPADE_4.png'
    anno = Image.open(one_image)
    anno = Image.fromarray(np.uint8(anno)).convert('RGB')
    # from_im = from_im.convert('RGB') if opts.label_nc == 0 else from_im.convert('L')
    anno, _ = RGBAnno2Mask(anno, mappingDict)
    annoTransform = getTransforms(mode='anno')
    anno = annoTransform(anno)*255
    # anno = transform(from_im)
    # print(from_im.shape)
    # exit(0)


    # from_im = Image.open(from_path)
    # from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
    # if self.transform:
    #     from_im = self.transform(from_im)

    # dataset = InferenceDataset(root=opts.data_path,
    #                            transform=transforms_dict['transform_inference'],
    #                            opts=opts)
    # dataloader = DataLoader(dataset,
    #                         batch_size=opts.test_batch_size,
    #                         shuffle=False,
    #                         num_workers=int(opts.test_workers),
    #                         drop_last=True)

    # if opts.n_images is None:
    #     opts.n_images = len(dataset)

    # global_i = 0
    # global_time = []
    # for input_batch in tqdm(dataloader):
    #     if global_i >= opts.n_images:
    #         break
    with torch.no_grad():
        # input_batch = input array
        anno = torch.unsqueeze(anno, 0)
        anno = anno.cuda()
        anno = convertAnnoTensor(anno, 1399)
        anno = anno.float()
        # tic = time.time()
        result_batch = run_on_batch(anno, net, opts)
        # toc = time.time()
        # global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            # im_path = dataset.paths[global_i]

            # if opts.couple_outputs or global_i % 100 == 0:
            #     input_im = log_input_image(input_batch[i], opts)
            #     resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
            #     if opts.resize_factors is not None:
            #         # for super resolution, save the original, down-sampled, and output
            #         source = Image.open(im_path)
            #         res = np.concatenate([np.array(source.resize(resize_amount)),
            #                               np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
            #                               np.array(result.resize(resize_amount))], axis=1)
            #     else:
            #         # otherwise, save the original and output
            #         res = np.concatenate([np.array(input_im.resize(resize_amount)),
            #                               np.array(result.resize(resize_amount))], axis=1)
            #     Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            # im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            # Image.fromarray(np.array(result)).save(im_save_path)
            img = Image.fromarray(np.array(result))
            img.show()

            # global_i += 1

    # stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    # result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    # print(result_str)

    # with open(stats_path, 'w') as f:
    #     f.write(result_str)


def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None: # temporary not to use style mixing
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == '__main__':
    run()
