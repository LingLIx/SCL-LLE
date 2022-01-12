
from torchvision import transforms as transforms
from tqdm import tqdm
import network
import utils
import random
import argparse
import numpy as np
from metrics import StreamSegMetrics
from torch.utils import data
from datasets import Cityscapes
from utils import ext_transforms as et
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import sys
import time
import lowlight_model
import Myloss
from torchvision import transforms
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from PIL import Image, ImageStat
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os,sys
import random
import shutil
import math

transf = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])



def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data/cityscapes',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=50e3,
                        help="epoch number")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=384)
    
    parser.add_argument("--ckpt", default="./checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='28333',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')

    parser.add_argument('--lowlight_lr', type=float, default=0.0001)
    parser.add_argument('--lowlight_weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--snapshots_folder', type=str, default="SCL-LLE/")
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            #et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 19
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1
    
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return


    #lowlight
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    DCE_net = lowlight_model.enhance_net_nopool().cuda()
    DCE_net.apply(weights_init)
    L_color = Myloss.L_color()
    L_con = Myloss.L_con()
    L_TV = Myloss.L_TV()
    L_segexp = Myloss.L_segexp()
    L_percept = Myloss.perception_loss()
    L_const = Myloss.PerceptualLoss()
    
    
    lowlight_optimizer = torch.optim.Adam(DCE_net.parameters(), lr=opts.lowlight_lr, weight_decay=opts.lowlight_weight_decay)

    pathL = "./datasets/data/Contrast/low/"  # negative samples
    pathDirL = os.listdir(pathL)
    pathGT = "./datasets/data/Contrast/GT/"  #positive samples
    pathDirGT = os.listdir(pathGT)
    picknumber = 2
    interval_loss = 0
    cnt = 1
    
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        DCE_net.train()
        model.train()
        cur_epochs += 1
        transff = transforms.ToTensor()
        for (images, labels) in train_loader:
            if len(images) <=1:
                continue
            cur_itrs += 1
            sampleL = random.sample(pathDirL, picknumber)         
            X_L = Image.open(pathL+sampleL[0])
            X_L = X_L.resize((384, 384),Image.ANTIALIAS) 
            X_L = transff(X_L)

            Y_L = Image.open(pathL+sampleL[1])
            Y_L = Y_L.resize((384, 384),Image.ANTIALIAS) 
            Y_L = transff(Y_L)
            L = torch.stack((X_L, Y_L), 0)
            L = L.cuda()

            sampleGT = random.sample(pathDirGT, picknumber)         
            X_GT = Image.open(pathGT+sampleGT[0])
            X_GT = X_GT.resize((384, 384),Image.ANTIALIAS) 
            X_GT = transff(X_GT)
            
            Y_GT = Image.open(pathGT+sampleGT[1])
            Y_GT = Y_GT.resize((384, 384),Image.ANTIALIAS) 
            Y_GT = transff(Y_GT)

            GT = torch.stack((X_GT, Y_GT), 0)
            GT = GT.cuda()
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
 
            img_lowlight = images.cuda()
            enhanced_image_1,enhanced_image,A  = DCE_net(images)
            
            Loss_TV = 200*L_TV(A)		
            loss_col = 8*torch.mean(L_color(enhanced_image))
            loss_segexp = torch.mean(L_segexp(enhanced_image, labels))
            loss_percent = torch.mean(L_percept(images,enhanced_image))
            loss_cont = 10*torch.mean(max(L_con(enhanced_image, GT) -L_con(enhanced_image,L) + 0.3 ,L_con(L, enhanced_image)-L_con(L, enhanced_image)))
            loss_cont2 = 5*torch.mean(max(L_const(enhanced_image, GT) -L_const(enhanced_image,L) + 0.04 ,L_con(L, enhanced_image)-L_con(L, enhanced_image)))
            


            enhanced_image = enhanced_image.detach()

            
            enhanced_image = enhanced_image.to(device, dtype = torch.float32)
            image1, image2 = enhanced_image.split(1,dim=0)
            image1 = image1.squeeze()
            image2 = image2.squeeze()
            image1 = transf(image1)
            image2 = transf(image2)
            images = torch.stack([image1, image2], dim = 0)
            
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
    
            seg_loss =torch.tensor(loss, device='cuda:0')
              
            lowlight_loss = Loss_TV+ loss_col+ loss_segexp +loss_cont + loss_cont2 + seg_loss+loss_percent

            
            lowlight_optimizer.zero_grad()
            lowlight_loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(),opts.grad_clip_norm)
            lowlight_optimizer.step()
    
            cnt=cnt+1

            if (cnt % 1300)==0:
                torch.save(DCE_net.state_dict(), opts.snapshots_folder + "Epoch" + str(cnt/1300) + '.pth')
       
            if cur_itrs >=  opts.total_itrs:
                return

        
if __name__ == '__main__':
    main()
