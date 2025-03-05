import os
import time
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from utils.eval_metrics import segment_level, event_level


from imagebind_finetune.models import imagebind_model_fully
from imagebind_finetune.models.imagebind_model_fully import ModalityType
from einops import rearrange, repeat, reduce

from dataloader_fully import OVAVE_Dataset
from config import cfg
import pdb


# configs
# dataset_configs = get_and_save_args(parser)
# parser.set_defaults(**dataset_configs)
# args = parser.parse_args()
args = get_and_save_args(parser)
# print parameters
# print('----------------args-----------------')
# for k in list(vars(args).keys()):
#     print('%s: %s' % (k, vars(args)[k]))
# print('----------------args-----------------')



 # =================================  seed config ============================
SEED = args.seed
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =============================================================================


# select GPUs
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

'''Create snapshot_pred dir for copying code and saving models '''
if not os.path.exists(args.snapshot_pref):
    os.makedirs(args.snapshot_pref, exist_ok=True)

if os.path.isfile(args.resume):
    args.snapshot_pref = os.path.dirname(args.resume)

logger = Prepare_logger(args, eval=args.evaluate)

if cfg.SA.SA_ATTN_FLAG:
    if cfg.SA.SA_ATTN_TYPE == 'bothFirst':
        ext = f"_Sattn_bothFirstK{cfg.SA.SA_BOTH_LAYER_K}"
        audio_layer_ids = list(range(cfg.SA.SA_BOTH_LAYER_K))
        visual_layer_ids = list(range(cfg.SA.SA_BOTH_LAYER_K))
        sattn_type = 'bothFirst'
    elif cfg.SA.SA_ATTN_TYPE == 'bothLast':
        ext = f"_Sattn_bothLastK{cfg.SA.SA_BOTH_LAYER_K}"
        audio_layer_ids = list(range(12-cfg.SA.SA_BOTH_LAYER_K, 12))
        visual_layer_ids = list(range(32-cfg.SA.SA_BOTH_LAYER_K, 32))
        sattn_type = 'bothLast'
    elif cfg.SA.SA_ATTN_TYPE == 'evenFirst':
        ext = f"_Sattn_evenly_firstK{cfg.SA.SA_EVEN_LAYER_K}"
        audio_layer_ids  = list(range(0, 12, 3)) + [11] # [0, 3, 6, 9, 11]
        visual_layer_ids = list(range(0, 32, 8)) + [31] # [0, 8, 16, 24, 31]
        audio_layer_ids = audio_layer_ids[:cfg.SA.SA_EVEN_LAYER_K]
        visual_layer_ids = visual_layer_ids[:cfg.SA.SA_EVEN_LAYER_K]
        sattn_type = 'evenFirst'
    elif cfg.SA.SA_ATTN_TYPE == 'evenLast':
        ext = f"_Sattn_evenly_lastK{cfg.SA.SA_EVEN_LAYER_K}"
        audio_layer_ids  = list(range(0, 12, 3)) + [11] # [0, 3, 6, 9, 11]
        visual_layer_ids = list(range(0, 32, 8)) + [31] # [0, 8, 16, 24, 31]
        audio_layer_ids = audio_layer_ids[-cfg.SA.SA_EVEN_LAYER_K:]
        visual_layer_ids = visual_layer_ids[-cfg.SA.SA_EVEN_LAYER_K:]
        sattn_type = 'evenLast'
    elif cfg.SA.SA_ATTN_TYPE == 'fixedBlkids':
        ext = f"_Sattn_fixedBlkids"
        audio_layer_ids = cfg.SA.SA_FIXED_LAYER_A
        visual_layer_ids = cfg.SA.SA_FIXED_LAYER_V
        sattn_type = 'fixedBlkids'
        assert len(audio_layer_ids) == len(visual_layer_ids)
    else:
        raise NotImplementedError
else:
    ext = "_noSattn"
    audio_layer_ids = []
    visual_layer_ids = []
    sattn_type = 'none'
# print(f"==> Spatial Attention Mode: {sattn_type}")
# print(f"==> Spatial Attention audio_layer_ids: {audio_layer_ids}")
# print(f"==> Spatial Attention visual_layer_ids: {visual_layer_ids}")
logger.info(f"==> Spatial Attention Mode: {sattn_type}")
logger.info(f"==> Spatial Attention audio_layer_ids: {audio_layer_ids}")
logger.info(f"==> Spatial Attention visual_layer_ids: {visual_layer_ids}")


if not args.evaluate:
    logger.info(f'\nCreating folder: {args.snapshot_pref}')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
else:
    logger.info(f'\nLog file will be save in {args.snapshot_pref}/Eval_{args.test_data_type}.log.')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))



def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(
        OVAVE_Dataset(split='train', debug=args.debug),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        OVAVE_Dataset(split='val', test_data_type=args.val_data_type, debug=args.debug),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        OVAVE_Dataset(split='test', test_data_type=args.test_data_type, debug=args.debug),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    # pdb.set_trace()
    
    ''' Model selection '''
    mainModel = imagebind_model_fully.imagebind_huge(
        pretrained=True,
        # spatial attention,
        spatial_av_attn_layer_ids=(audio_layer_ids, visual_layer_ids),
        sattn_flag=sattn_type,
        # temporal attention,
        tattn_flag=cfg.TA.TA_ATTN_FLAG,
        sa_layer_num=cfg.TA.SA_LAYER_NUM,
        xa_layer_num=cfg.TA.XA_LAYER_NUM,
        hid_dim=cfg.TA.HIDDEN_DIM,
        d_ff=cfg.TA.FF_DIM,
        head_num=cfg.TA.HEAD_NUM,
        dropout=cfg.TA.DROPOUT,
        use_adj_in_attn=cfg.TA.USE_ADJ_IN_ATTN,
        gamma=cfg.TA.GAMMA,
        bias=cfg.TA.BIAS,
        use_mask_in_attn=cfg.TA.USE_MASK_IN_ATTN,
        win_size=cfg.TA.WIN_SIZE,
        norm_flag=cfg.TA.NORM_FLAG,
        # text projection tuning
        text_tune_flag=cfg.TEXT_TUNE_FLAG,
    )
    # mainModel.eval()
    mainModel.to(device)

    '''optimizer setting'''
    # mainModel = nn.DataParallel(mainModel).cuda()
    # learned_parameters = []
    for name, params in mainModel.named_parameters():
        if 'spatial_av_layers' in name:
            params.requires_grad = True
        elif 'temporal_av_layer' in name:
            params.requires_grad = True
        elif 'task_res' in name:
            params.requires_grad = True
            # learned_parameters.append(params)
        else:
            params.requires_grad = False
    for name, params in mainModel.named_parameters():
        if params.requires_grad:
            print(name)


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mainModel.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    """loss"""
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError


    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart testing..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return


    best_accuracy, best_accuracy_epoch = 0, 0
    '''Training and Evaluation'''
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)

        if ((epoch) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            # acc = validate_epoch(mainModel, val_dataloader, criterion, criterion_event, epoch)
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch) # may change to test
            if acc > best_accuracy:
                best_accuracy = acc
                logger.info(f'best average result of three metrics at epoch-{epoch}: {best_accuracy:.4f}')
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='FullySupervised',
                    epoch=epoch + 1,
                    seed=SEED
                )
        scheduler.step()


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    train_seg_fscore = AverageMeter()
    train_eve_fscore = AverageMeter()
    end_time = time.time()

    model.train()
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.
    # model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        audio, visual, text, full_label, avc_label, category_label, vid_name = batch_data
        audio_inputs = audio.squeeze(1).cuda()  # [bs, 10, 1, 128, 204]
        visual_inputs = visual.cuda() # [bs, 10, 3, 224, 224]
        text_inputs = text[0].cuda() # [46+1, 77]
        avc_labels = avc_label.cuda() # [bs, 10] 0 or 1
        full_labels = full_label.cuda() # [bs, 10] category id for CE loss
        category_labels = category_label.squeeze(1).cuda() # [bs]
        

        """forward"""
        bs = visual_inputs.size(0)
        inputs = {
            ModalityType.TEXT: text_inputs.cuda(), # [46+1, 77]
            ModalityType.VISION: visual_inputs.cuda(), # [bs, 10, 3, 224, 224]
            ModalityType.AUDIO: audio_inputs.cuda(), # [bs, 10, 1, 128, 204]
        }
        # with torch.no_grad():
        embeddings = model(inputs) #todo
        audio_feas = embeddings['audio'] # [bs, 10, 1024], without normalization
        visual_feas = embeddings['vision'] # [bs, 10, 1024], without normalization
        text_feas = embeddings['text'] # [46+1, 1024], without normalization
        text_feas = text_feas.unsqueeze(0).repeat(bs, 1, 1) # [bs, 46+1, 1024]
        # pred_logits = embeddings['pred'] # [bs, 10, 46+1], without sigmoid


        #! GEN1: generate AVE prediction logits
        simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 46+1]
        simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 46+1]
        pred_logits = torch.sqrt(F.relu(simm_at * simm_vt)) # [bs, 10, 46+1]
        # min-max normalization
        x_min = (pred_logits.min(-1)[0]).unsqueeze(-1) # [bs, 10, 1]
        x_max = (pred_logits.max(-1)[0]).unsqueeze(-1) # [bs, 10, 1]
        # pred_logits = (pred_logits - x_min) / (x_max - x_min)
        pred_logits = (pred_logits - x_min) / (x_max - x_min + 1e-8) # for spatialand temoral
        # pdb.set_trace()
        #! GEN2: generate AVE prediction logits
        # simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 46+1]
        # simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 46+1]
        # pred_logits = (simm_at + simm_vt) * 0.5 # [bs, 10, 46+1]
        #! GEN3: generate AVE prediction logits
        # av_feas = (audio_feas + visual_feas) * 0.5
        # simm_avt = compute_cross_modal_similarity(av_feas, text_feas) # [bs, 10, 46+1]
        # pred_logits = simm_avt # [bs, 10, 46+1]


        if  torch.sum(pred_logits.isnan()) > 0:
            continue
            pdb.set_trace()


        loss = criterion_event(pred_logits.permute(0, 2, 1), full_labels)
        loss.backward()


        '''Compute Accuracy'''
        if args.test_strategy_type == 'v01':
            # simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 46+1]
            # simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 46+1]
            is_event_scores, event_scores = postprocess_simm(simm_at, simm_vt, bg_flag=cfg.TRAIN_BG_CLASS_ID) # [bs, 10], [bs, 46+1]
            acc = compute_accuracy_supervised_v0(is_event_scores, event_scores, avc_labels, category_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_acc.update(acc.item(), bs * 10)
            # pdb.set_trace()
            seg_f, eve_f = compute_seg_eve_fscores_v0(is_event_scores, event_scores, avc_labels, category_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_seg_fscore.update(seg_f, n=1)
            train_eve_fscore.update(eve_f, n=1)
        elif args.test_strategy_type == 'v0':
            simm_at = simm_at[:, :, :-1] # [bs, 10, 46]
            simm_vt = simm_vt[:, :, :-1] # [bs, 10, 46]
            is_event_scores, event_scores = postprocess_simm(simm_at, simm_vt, bg_flag=cfg.TRAIN_BG_CLASS_ID) # [bs, 10], [bs, 46+1]
            acc = compute_accuracy_supervised_v0(is_event_scores, event_scores, avc_labels, category_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_acc.update(acc.item(), bs * 10)
            # pdb.set_trace()
            seg_f, eve_f = compute_seg_eve_fscores_v0(is_event_scores, event_scores, avc_labels, category_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_seg_fscore.update(seg_f, n=1)
            train_eve_fscore.update(eve_f, n=1)
        elif args.test_strategy_type == 'v1':
            acc = compute_accuracy_supervised_v1(pred_logits, full_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_acc.update(acc.item(), bs*10)
            seg_f, eve_f = compute_seg_eve_fscores_v1(pred_logits, avc_labels, category_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_seg_fscore.update(seg_f, n=1)
            train_eve_fscore.update(eve_f, n=1)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            # total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient) 
            total_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.clip_gradient) 
            # if total_norm > args.clip_gradient:
            #     logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()


        losses.update(loss.item(), bs*10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()


        '''Print logs in Terminal'''
        if n_iter % args.print_iter_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg:.3f})\t'
                f'Seg@F1 {train_seg_fscore.val:.3f} ({train_seg_fscore.avg:.3f})\t'
                f'Eve@F1 {train_eve_fscore.val:.3f} ({train_eve_fscore.avg:.3f})'
            )
    return losses.avg



def compute_cross_modal_similarity(tensor_a, tensor_t):
    B, T, D = tensor_a.shape
    _, C, _ = tensor_t.shape
    tensor_a_expanded = tensor_a.unsqueeze(2).expand(B, T, C, D)
    tensor_t_expanded = tensor_t.unsqueeze(1).expand(B, T, C, D)
    cos_sim = F.cosine_similarity(tensor_a_expanded, tensor_t_expanded, dim=-1)
    
    # norm_a = F.normalize(tensor_a, dim=-1)
    # norm_t = F.normalize(tensor_t, dim=-1)
    # simm_at = torch.bmm(norm_a, norm_t.permute(0, 2, 1))
    # pdb.set_trace()
    return cos_sim

def postprocess_simm(simm_at, simm_vt, bg_flag=cfg.TOTAL_BG_CLASS_ID):
    # simm_at, simm_vt: [B, T, C=67 / 67+1]
    max_prob_at_idx = simm_at.max(dim=-1)[1] # [B, T]
    max_prob_vt_idx = simm_vt.max(dim=-1)[1] # [B, T]
    is_event_flag = (max_prob_at_idx == max_prob_vt_idx).float() # [B, T]
    # is_event_flag = (simm_at.max(dim=-1)[1] == simm_vt.max(dim=-1)[1]).float() # [B, T]
    B = is_event_flag.shape[0]
    # C = simm_at.shape[-1]
    event_flag = torch.zeros([B, bg_flag+1]).cuda()
    # print(is_event_flag)
    for i in range(B):
        if torch.all(is_event_flag[i] == 0):
            # event_flag[i][cfg.TOTAL_BG_CLASS_ID] = 1 # background video, all segments are backgrounds
            event_flag[i][bg_flag] = 1 # background video, all segments are backgrounds
        else: 
            nonzero_pos = torch.nonzero(is_event_flag[i], as_tuple=False).squeeze()
            if len(nonzero_pos.size()) == 0: # only one postive position
                category_id = max_prob_at_idx[i][nonzero_pos.item()]
            else:
                category_id = max_prob_at_idx[i][nonzero_pos[0].item()]
            event_flag[i][category_id] = 1 
    # pdb.set_trace()
    return is_event_flag, event_flag


@torch.no_grad()
def validate_epoch(model, val_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    seg_fscore = AverageMeter()
    eve_fscore = AverageMeter()
    end_time = time.time()

    model.eval()


    for n_iter, batch_data in enumerate(val_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        audio, visual, text, full_label, avc_label, category_label, vid_name = batch_data
        audio_inputs = audio.squeeze(1).cuda()  # [bs, 10, 1, 128, 204]
        visual_inputs = visual.cuda() # [bs, 10, 3, 224, 224]
        text_inputs = text[0].cuda() # [46/67, 77]
        avc_labels = avc_label.cuda() # [bs, 10], 0 or 1
        category_labels = category_label.squeeze(1).cuda() # [bs]
        full_labels = full_label.cuda() # [bs, 10], class ids
        

        bs = visual_inputs.size(0)
        inputs = {
            ModalityType.TEXT: text_inputs.cuda(), # [67+1, 77]
            ModalityType.VISION: visual_inputs.cuda(), # [bs, 10, 3, 224, 224]
            ModalityType.AUDIO: audio_inputs.cuda(), # [bs, 10, 1, 128, 204]
        }
        with torch.no_grad():
            embeddings = model(inputs)
        audio_feas = embeddings['audio'] # [bs, 10, 1024]
        visual_feas = embeddings['vision'] # [bs, 10, 1024]
        text_feas = embeddings['text'] # [67+1, 1024]
        text_feas = text_feas.unsqueeze(0).repeat(bs, 1, 1) # [bs, 67+1, 1024]
        # pred_logits = embeddings['pred'] # [bs, 10, 67+1]

        #! GEN1: generate AVE prediction logits
        simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 46+1]
        simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 46+1]
        pred_logits = torch.sqrt(F.relu(simm_at * simm_vt)) # [bs, 10, 46+1]
        # min-max normalization
        x_min = (pred_logits.min(-1)[0]).unsqueeze(-1) # [bs, 10, 1]
        x_max = (pred_logits.max(-1)[0]).unsqueeze(-1) # [bs, 10, 1]
        # pred_logits = (pred_logits - x_min) / (x_max - x_min)
        pred_logits = (pred_logits - x_min) / (x_max - x_min + 1e-8) # for spatial and temoral
        # pdb.set_trace()
        #! GEN2: generate AVE prediction logits
        # simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 46+1]
        # simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 46+1]
        # pred_logits = (simm_at + simm_vt) * 0.5 # [bs, 10, 46+1]
        #! GEN3: generate AVE prediction logits
        # av_feas = (audio_feas + visual_feas) * 0.5
        # simm_avt = compute_cross_modal_similarity(av_feas, text_feas) # [bs, 10, 46+1]
        # pred_logits = simm_avt # [bs, 10, 46+1]


        # baseline v0: training free: computing audio-text visual-text similarity
        if args.test_strategy_type != 'v1':
            if args.test_strategy_type == 'v0':
                text_feas = text_feas[:, :-1, :] # [bs, 67, 1024]
                # pdb.set_trace()
                simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 67]
                simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 67]
                # pdb.set_trace()
                is_event_scores, event_scores = postprocess_simm(simm_at, simm_vt, bg_flag=cfg.TOTAL_BG_CLASS_ID) # [bs, 10], [bs, 67+1]
                # is_event_scores: avc predction; 
                # event_scores: event category prediction
            # baseline v01: training free: using `other' category, computing audio-text visual-text similarity"""
            elif args.test_strategy_type == 'v01':
                # pdb.set_trace()
                # simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 67+1]
                # simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 67+1]
                is_event_scores, event_scores = postprocess_simm(simm_at, simm_vt, bg_flag=cfg.TOTAL_BG_CLASS_ID) # [bs, 10], [bs, 67+1]
            loss_is_event = criterion_event(is_event_scores, avc_labels)
            loss_event_class = criterion_event(event_scores, category_labels)
            loss = loss_is_event + loss_event_class

            # print(vid_name)
            acc = compute_accuracy_supervised_v0(is_event_scores, event_scores, avc_labels, category_labels)
            accuracy.update(acc.item(), bs * 10)
            # pdb.set_trace()
            seg_f, eve_f = compute_seg_eve_fscores_v0(is_event_scores, event_scores, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID)
            seg_fscore.update(seg_f, n=1)
            eve_fscore.update(eve_f, n=1)
        else:
            loss = criterion_event(pred_logits.permute(0, 2, 1), full_labels)
            acc = compute_accuracy_supervised_v1(pred_logits, full_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID)
            accuracy.update(acc.item(), bs * 10)
            seg_f, eve_f = compute_seg_eve_fscores_v1(pred_logits, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID)
            seg_fscore.update(seg_f, n=1)
            eve_fscore.update(eve_f, n=1)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

        '''Print logs in Terminal'''
        # if n_iter % args.print_iter_freq == 0:
        if n_iter % args.print_iter_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(val_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                f'Seg@F1 {seg_fscore.val:.3f} ({seg_fscore.avg:.3f})\t'
                f'Eve@F1 {eve_fscore.val:.3f} ({eve_fscore.avg:.3f})'
            )

    logger.info(
        f"\tEvaluation results (acc): {accuracy.avg:.4f}\t (Segment-level F1score): {seg_fscore.avg:.4f}\t (Event-level F1score): {eve_fscore.avg:.4f}"
    )
    return (accuracy.avg + seg_fscore.avg + eve_fscore.avg) / 3




def compute_accuracy_supervised_v0(is_event_scores_pred, event_scores_pred, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID):
    # is_event_scores_pred, avc_labels: [bs, 10]
    # event_scores_pred: [bs, C+1]
    # category_labels: [bs]
    def obtain_label_mat(is_event_scores, event_scores, bg_flag):
        scores_mask = (is_event_scores == 0)
        _, event_class = event_scores.max(-1) # foreground classification, [B]
        pred = is_event_scores.long() # [B, 10]
        pred *= event_class[:, None]
        # add mask
        pred[scores_mask] = bg_flag # 67 denotes bg
        return pred
    
    pred = obtain_label_mat(is_event_scores_pred, event_scores_pred, bg_flag)
    is_event_scores_gt = avc_labels
    event_scores_gt = torch.zeros_like(event_scores_pred) # [bs, C+1]
    for i in range(event_scores_gt.size(0)): 
        event_scores_gt[i, category_labels[i]] = 1

    targets = obtain_label_mat(is_event_scores_gt, event_scores_gt, bg_flag)
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    # acc = correct_num * (100. / correct.numel())
    acc = correct_num / correct.numel()
    # pdb.set_trace()
    return acc

# def compute_accuracy_supervised(is_event_scores_pred, event_scores_pred, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID):
def compute_accuracy_supervised_v1(pred_logits, full_labels, bg_flag):
    # pred_logits: [bs, 10, 46/67 + 1]
    # full_labels: [bs, 10], class ids
    _, pred = pred_logits.max(-1)
    targets = full_labels
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    # acc = correct_num * (100. / correct.numel())
    acc = correct_num / correct.numel()
    # pdb.set_trace()
    return acc


def compute_seg_eve_fscores_v0(is_event_scores_pred, event_scores_pred, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID):
    # is_event_scores_pred, avc_labels: [bs, 10]
    # event_scores_pred: [bs, C+1]
    # category_labels: [bs]
    def obtain_pred_mat(is_event_scores, event_scores, bg_flag):
        B, T = is_event_scores.shape
        K = event_scores.size(-1) # C+1
        assert K == bg_flag + 1
        # pdb.set_trace()
        pred_mat = torch.zeros(B, K, T).cuda()
        for i in range(B):
            class_id = torch.nonzero(event_scores[i]).squeeze().item()
            if class_id != bg_flag:
                pred_mat[i][class_id] = is_event_scores[i]
                pred_mat[i][-1] = 1 - is_event_scores[i] #  background
            else:
                pred_mat[i][-1] = 1 - is_event_scores[i] #  background
            # pdb.set_trace()
        return pred_mat.cpu().data.numpy()
    
    def obtain_gt_mat(avc_labels, category_labels, bg_flag):
        # avc_labels: [bs, 10], category_labels: [bs]
        B, T = avc_labels.shape
        K = bg_flag + 1
        gt_mat = torch.zeros(B, K, T).cuda()
        for i in range(B):
            class_id = category_labels[i].item()
            if class_id != bg_flag:
                gt_mat[i][class_id] = avc_labels[i]
                gt_mat[i][-1] = 1 - avc_labels[i]
            else:
                gt_mat[i][-1] = 1 - avc_labels[i]
            # pdb.set_trace()
        return gt_mat.cpu().data.numpy()

    pred = obtain_pred_mat(is_event_scores_pred, event_scores_pred, bg_flag)
    targets = obtain_gt_mat(avc_labels, category_labels, bg_flag)
    # pdb.set_trace()
    
    B = avc_labels.shape[0]
    seg_fscore = np.zeros(B)
    eve_fscore = np.zeros(B)
    for i in range(B):
        seg_f = segment_level(pred[i], targets[i])
        seg_fscore[i] = seg_f
        eve_f = event_level(pred[i], targets[i])
        eve_fscore[i] = eve_f
    avg_batch_seg_fscore = np.mean(seg_fscore)
    avg_batch_eve_fscore = np.mean(eve_fscore)
    # pdb.set_trace()
    return avg_batch_seg_fscore, avg_batch_eve_fscore
    


def compute_seg_eve_fscores_v1(pred_logits, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID):
    # pred_logits: [bs, 10, 67+1]
    # avc_labels: [bs, 10], 0 or 1
    # category_labels: [bs]
    def obtain_pred_mat(pred_logits, bg_flag):
        pred_logits = pred_logits.permute(0, 2, 1) # [bs, 67+1, 10]
        assert pred_logits.shape[1] == bg_flag + 1
        pred_mat = torch.zeros_like(pred_logits).cuda() # [bs, 67+1, 10]
        pred_cls_ids = pred_logits.max(dim=1)[1] # [bs, 10]
        B = pred_logits.shape[0]
        for i in range(B):
            class_id = pred_cls_ids[i] # [10]
            for j in range(len(class_id)):
                pred_mat[i][class_id[j]][j] = 1
        return pred_mat.cpu().data.numpy()
    
    def obtain_gt_mat(avc_labels, category_labels, bg_flag):
        # avc_labels: [bs, 10], category_labels: [bs]
        B, T = avc_labels.shape
        K = bg_flag + 1
        gt_mat = torch.zeros(B, K, T).cuda()
        for i in range(B):
            class_id = category_labels[i].item()
            if class_id != bg_flag:
                gt_mat[i][class_id] = avc_labels[i]
                gt_mat[i][-1] = 1 - avc_labels[i]
            else:
                gt_mat[i][-1] = 1 - avc_labels[i]
            # pdb.set_trace()
        return gt_mat.cpu().data.numpy()

    pred = obtain_pred_mat(pred_logits, bg_flag) # [bs, 46/67 + 1, 10]
    targets = obtain_gt_mat(avc_labels, category_labels, bg_flag) # [bs, 46/67 + 1, 10]
    
    B = avc_labels.shape[0]
    seg_fscore = np.zeros(B)
    eve_fscore = np.zeros(B)
    for i in range(B):
        seg_f = segment_level(pred[i], targets[i])
        seg_fscore[i] = seg_f
        eve_f = event_level(pred[i], targets[i])
        eve_fscore[i] = eve_f
    avg_batch_seg_fscore = np.mean(seg_fscore)
    avg_batch_eve_fscore = np.mean(eve_fscore)
    # pdb.set_trace()
    return avg_batch_seg_fscore, avg_batch_eve_fscore
    


def save_checkpoint(state_dict, top1, task, epoch, seed):
    # model_name = f'{args.snapshot_pref}/model_seed_{seed}_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    model_name = f'{args.snapshot_pref}/task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)
    print("best model is saved to ", model_name)


if __name__ == '__main__':
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f'current time: {cur}')
    start = time.time()
    
    main()

    end = time.time()
    print(f'duration time {(end - start) / 60} mins.')
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f'current time: {cur}')