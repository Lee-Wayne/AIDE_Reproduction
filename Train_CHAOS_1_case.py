import sys
import os, time, argparse, random
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import PolynomialLR
import pandas as pd
import torch.nn.functional as F
from skimage import measure
from AIDE_Reproduction.model import FuseUNet
from dataset_CHAOS_pre_process import CHAOS_seg, Compose, Resize, ToTensor, Normalize
from utils import CrossEntropyLoss2d, DiceLoss, MultiClassDiceLoss, CEMDiceLoss, PolyLR, Dice_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Segmeantation for CHAOS',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default='fuseunet', type=str, help='fuseunet, ...')
    parser.add_argument('--data_mean', default=None, nargs='+', type=float,
                        help='Normalize mean')
    parser.add_argument('--data_std', default=None, nargs='+', type=float,
                        help='Normalize std')
    parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
    parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
    parser.add_argument('--gpu_order', default='0', type=str, help='gpu order')
    parser.add_argument('--torch_seed', default=2, type=int, help='torch_seed')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=100, type=int, help='num epoch')
    parser.add_argument('--loss', default='cedice', type=str, help='ce, dice')
    parser.add_argument('--img_size', default=256, type=int, help='512')
    parser.add_argument('--lr_policy', default='StepLR', type=str, help='StepLR')
    parser.add_argument('--cedice_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='weight for ce and dice loss')
    parser.add_argument('--ceclass_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='categorical weight for ce loss')
    parser.add_argument('--diceclass_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='categorical weight for dice loss')
    parser.add_argument('--checkpoint', default='checkpoint_chaos_comparison1case/')
    parser.add_argument('--history', default='history_chaos_comparison1case')
    parser.add_argument('--cudnn', default=0, type=int, help='cudnn')
    parser.add_argument('--repetition', default=2, type=int, help='...')

    args = parser.parse_args()
    return args


def record_params(args):
    localtime = time.asctime(time.localtime(time.time()))
    logging.info('Segmeantation for CHAOS MR(Data: {}) \n'.format(localtime))
    logging.info('**************Parameters***************')

    args_dict = args.__dict__
    for key, value in args_dict.items():
        logging.info('{}: {}'.format(key, value))
    logging.info('**************Parameters***************\n')


def keep_largest_connected_components(mask):
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    blobs = measure.label(mask, connectivity=1)  # connectivity 1: 4 neighbours 2: 8 neighbours
    props = measure.regionprops(blobs)
    area = [ele.area for ele in props]
    if mask.max() > 0:
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_img[blobs == largest_blob_label] = 1
    return out_img


def Dice3d_fn(inputs, targets):
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    intersection = iflat * tflat
    intersection = 2 * np.sum(intersection)
    union = np.sum(iflat) + np.sum(tflat)
    dice_image = intersection / union
    return dice_image


def build_model(model_name, num_classes):
    if model_name == 'fuseunet':
        net = FuseUNet(num_classes=num_classes)
    else:
        raise ValueError('Model not implemented')
    return net


def Train(train_root, train_csv, test_csv, traincase_csv, testcase_csv):
    args = parse_args()

    train_cases = pd.read_csv(traincase_csv)['patient_case'].tolist()
    test_cases = pd.read_csv(testcase_csv)['patient_case'].tolist()

    record_params(args)

    # set seed
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_order
    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)
    np.random.seed(args.torch_seed)
    random.seed(args.torch_seed)

    if args.cudnn == 0:
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
        cudnn.deterministic = True

    num_classes = 2

    params_name = f'{args.model_name}_{args.repetition}.pkl'
    start_epoch = 0
    end_epoch = start_epoch + args.num_epoch

    history = {'train_loss': [], 'test_loss': [],
               'train_dice': [], 'test_dice': []}

    # build net
    net = build_model(args.model_name, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f'We are using {torch.cuda.device_count()} GPUs')
        net = nn.DataParallel(net)

    net.to(device)

    # Data Processing
    img_size = args.img_size
    ## Train
    train_aug = Compose([Resize(size=(img_size, img_size)),
                         ToTensor(),
                         Normalize(mean=args.data_mean, std=args.data_std)])
    ## Test
    test_aug = train_aug

    train_dataset = CHAOS_seg(root=train_root, csv_file=train_csv, transform=train_aug)
    test_dataset = CHAOS_seg(root=train_root, csv_file=test_csv, transform=test_aug)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False)

    # loss function, optimizer & lr scheduler
    cedice_weight = torch.tensor(args.cedice_weight)
    ceclass_weight = torch.tensor(args.ceclass_weight)
    diceclass_weight = torch.tensor(args.diceclass_weight)

    if args.loss == 'ce':
        criterion = CrossEntropyLoss2d(weight=ceclass_weight).to(device)
    elif args.loss == 'dice':
        criterion = MultiClassDiceLoss(weight=diceclass_weight).to(device)
    elif args.loss == 'cedice':
        criterion = CEMDiceLoss(cediceweight=cedice_weight, ceclassweight=ceclass_weight,
                                diceclassweight=diceclass_weight).to(device)
    else:
        print('Do not have this loss')

    optimizer = Adam(net.parameters(), lr=args.lr, amsgrad=True)

    if args.lr_policy == 'StepLR':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    if args.lr_policy == 'PolyLR':
        # scheduler = PolyLR(optimizer, max_epoch=end_epoch, power=0.9)
        # TODO: 尝试用pytorch的polyLR
        scheduler = PolynomialLR(optimizer, total_iters=100, power=0.9)

    # training process
    logging.info('Start Training For CHAOS Seg')
    besttraincasedice = 0.0
    besttestcasedice = 0.0

    for epoch in range(start_epoch, end_epoch):
        ts = time.time()

        # train
        net.train()
        train_loss = 0.
        train_dice = 0.
        train_count = 0.

        for batch_idx, (inphase, outphase, _, targets) in \
                tqdm(enumerate(train_loader), total=int(len(train_loader.dataset) / args.batch_size)):
            inphase = inphase.to(device)
            outphase = outphase.to(device)
            targets = targets[:, 1, :, :].to(device)
            optimizer.zero_grad()
            outputs = net(inphase, outphase)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inphase.size(0)
            train_dice += Dice_fn(outputs, targets)
            train_count += inphase.size(0)

        train_epoch_loss = train_loss / train_count
        train_epoch_dice = train_dice / train_count
        history['train_loss'].append(train_epoch_loss)
        history['train_dice'].append(train_epoch_dice)

        # test
        net.eval()
        test_loss = 0.
        test_dice = 0.
        test_count = 0.

        for batch_idx, (inphase, outphase, _, targets) in \
                tqdm(enumerate(test_loader), total=int(len(test_loader.dataset) / args.batch_size)):
            inphase = inphase.to(device)
            outphase = outphase.to(device)
            targets = targets[:, 1, :, :].to(device)
            with torch.no_grad():
                outputs = net(inphase, outphase)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inphase.size(0)
                dice = Dice_fn(outputs, targets)
            test_dice += dice
            test_count += inphase.shape[0]

        test_epoch_loss = test_loss / test_count
        test_epoch_dice = test_dice / test_count
        history['test_loss'].append(test_epoch_loss)
        history['test_dice'].append(test_epoch_dice)

        # Cross-model self-correcting

        testcasedices = torch.zeros(len(test_cases))
        startimgslices = torch.zeros(len(test_cases))
        for casecount in tqdm(range(len(test_cases)), total=len(test_cases)):
            caseidx = test_cases[casecount]
            caseinphaseimg = [file for file in test_dataset.inphase if int(file.split('/')[0]) == caseidx]
            caseinphaseimg.sort()
            caseoutphaseimg = [file for file in test_dataset.outphase if int(file.split('/')[0]) == caseidx]
            caseoutphaseimg.sort()
            casemask = [file for file in test_dataset.masks if int(file.split('/')[0]) == caseidx]
            casemask.sort()
            generatedtarget = []
            target = []
            startcaseimg = int(torch.sum(startimgslices[:casecount + 1]))
            for imgidx in range(len(caseinphaseimg)):
                assert caseinphaseimg[imgidx].split('/')[-1].split('.')[0] == \
                       casemask[imgidx].split('/')[-1].split('.')[0]
                assert caseinphaseimg[imgidx].split('/')[-1].split('-')[1] == \
                       caseoutphaseimg[imgidx].split('/')[-1].split('-')[1]
                assert int(caseinphaseimg[imgidx].split('/')[-1].split('-')[-1].split('.')[0]) == \
                       int(caseoutphaseimg[imgidx].split('/')[-1].split('-')[-1].split('.')[0]) + 1
                sample = test_dataset.__getitem__(imgidx + startcaseimg)
                inphase = sample[0]
                outphase = sample[1]
                mask = sample[3]
                target.append(mask[1, :, :])
                with torch.no_grad():
                    inphase = torch.unsqueeze(inphase.to(device), 0)
                    outphase = torch.unsqueeze(outphase.to(device), 0)
                    output = net(inphase, outphase)
                    output = F.softmax(output, dim=1)
                    output = torch.argmax(output, dim=1)
                    output = output.squeeze().cpu().numpy()
                    generatedtarget.append(output)
            target = np.stack(target, axis=-1)
            generatedtarget = np.stack(generatedtarget, axis=-1)
            generatedtarget_keeplargest = keep_largest_connected_components(generatedtarget)
            testcasedices[casecount] = Dice3d_fn(generatedtarget_keeplargest, target)
            if casecount + 1 < len(test_cases):
                startimgslices[casecount + 1] = len(caseinphaseimg)
        testcasedice = testcasedices.sum() / float(len(test_cases))

        traincasedices = torch.zeros(len(train_cases))
        startimgslices = torch.zeros(len(train_cases))
        generatedmask = []
        for casecount in tqdm(range(len(train_cases)), total=len(train_cases)):
            caseidx = train_cases[casecount]
            caseinphaseimg = [file for file in train_dataset.inphase if int(file.split('/')[0]) == caseidx]
            caseinphaseimg.sort()
            caseoutphaseimg = [file for file in train_dataset.outphase if int(file.split('/')[0]) == caseidx]
            caseoutphaseimg.sort()
            casemask = [file for file in train_dataset.masks if file.split('/')[0].isdigit()]
            casemask = [file for file in casemask if int(file.split('/')[0]) == caseidx]
            casemask.sort()
            generatedtarget = []
            target = []
            startcaseimg = int(torch.sum(startimgslices[:casecount + 1]))
            for imgidx in range(len(caseinphaseimg)):
                assert caseinphaseimg[imgidx].split('/')[-1].split('.')[0] == \
                       casemask[imgidx].split('/')[-1].split('.')[0]
                assert caseinphaseimg[imgidx].split('/')[-1].split('-')[1] == \
                       caseoutphaseimg[imgidx].split('/')[-1].split('-')[1]
                assert int(caseinphaseimg[imgidx].split('/')[-1].split('-')[-1].split('.')[0]) == \
                       int(caseoutphaseimg[imgidx].split('/')[-1].split('-')[-1].split('.')[0]) + 1
                sample = train_dataset.__getitem__(imgidx + startcaseimg)
                inphase = sample[0]
                outphase = sample[1]
                mask = sample[3]
                target.append(mask[1, :, :])
                with torch.no_grad():
                    inphase = torch.unsqueeze(inphase.to(device), 0)
                    outphase = torch.unsqueeze(outphase.to(device), 0)
                    output = net(inphase, outphase)
                    output = F.softmax(output, dim=1)
                    output = torch.argmax(output, dim=1)
                    output = output.squeeze().cpu().numpy()
                    generatedtarget.append(output)
            target = np.stack(target, axis=-1)
            generatedtarget = np.stack(generatedtarget, axis=-1)
            generatedtarget_keeplargest = keep_largest_connected_components(generatedtarget)
            traincasedices[casecount] = Dice3d_fn(generatedtarget_keeplargest, target)
            generatedmask.append(generatedtarget_keeplargest)
            if casecount + 1 < len(train_cases):
                startimgslices[casecount + 1] = len(caseinphaseimg)
        traincasedice = traincasedices.sum() / float(len(train_cases))

        time_cost = time.time() - ts

        logging.info(
            'epoch[%d/%d]: train_loss: %.3f | test_loss: %.3f | train_dice: %.3f | test_dice: %.3f || time: %.1f'
            % (epoch + 1, end_epoch, train_epoch_loss, test_epoch_loss, train_epoch_loss, test_epoch_dice, time_cost))
        logging.info(
            'epoch[%d/%d]: traincase_dice: %.3f | testcase_dice: %.3f || time: %.1f'
            % (epoch + 1, end_epoch, traincasedice, testcasedice, time_cost))

        if args.lr_policy != 'None':
            scheduler.step()

        if traincasedice > besttraincasedice or \
                (traincasedice == besttraincasedice and testcasedice > besttestcasedice):
            besttraincasedice = traincasedice
            besttestcasedice = testcasedice
            logging.info('Best Checkpoint {} Saving...'.format(epoch + 1))

            save_model = net
            if torch.cuda.device_count() > 1:
                save_model = list(net.children())[0]
            state = {
                'net': save_model.state_dict(),
                'loss': test_epoch_loss,
                'dice': test_epoch_dice,
                'epoch': epoch + 1,
                'history': history
            }
            save_check_name = os.path.join(args.checkpoint, params_name.split('.pkl')[0] +
                                         '_best_train_case_dice.' + params_name.split('.')[-1])
            torch.save(state, save_check_name)


args = parse_args()
if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)
if not os.path.exists(args.history):
    os.mkdir(args.history)

log_name = '{}_r{}.log'.format(args.model_name, args.repetition)
logging_save = os.path.join(args.history, log_name)
logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(logging_save)
                    ])

if __name__ == '__main__':
    train_root = 'input_chaos/All_Sets'
    train_csv = 'input_chaos/splitimages_cleanlabel/train_data_1cases.csv'
    test_csv = 'input_chaos/splitimages_cleanlabel/val_data_10cases.csv'
    traincase_csv = 'input_chaos/splitcases/train_data_1cases.csv'
    testcase_csv = 'input_chaos/splitcases/val_data_10cases.csv'
    Train(train_root, train_csv, test_csv, traincase_csv, testcase_csv)
