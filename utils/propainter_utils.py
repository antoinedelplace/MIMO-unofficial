import sys
sys.path.append(".")
sys.path.append("../ProPainter")

import torch
import os, cv2, tqdm
import numpy as np
import scipy

from utils.video_utils import frame_gen_from_video

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from inference_propainter import get_ref_index

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def load_models(use_half = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_folder = "../checkpoints/"

    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                    model_dir=checkpoint_folder, progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)
    
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                    model_dir=checkpoint_folder, progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                    model_dir=checkpoint_folder, progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()

    if use_half:
        fix_flow_complete = fix_flow_complete.half()
        model = model.half()

    return fix_raft, fix_flow_complete, model

def to_tensor(np_array):
    return torch.from_numpy(np_array).permute(0, 3, 1, 2).contiguous().float().div(255)

def read_frame_from_videos(frame_gen):
    frames = []
    for frame in frame_gen:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    return np.array(frames)

def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask

def infer_mask(frames, mask_dilates=4):
    threshold = 0.1*255
    n_batch, w, h, _ = np.shape(frames)
    masks_dilated = np.zeros((n_batch, w, h, 1), dtype=np.uint8)
          
    for i_frame, frame in enumerate(frames):
        mask_img = np.all(frame < threshold, axis=-1).astype(np.uint8)
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)

        masks_dilated[i_frame, :, :, 0] = mask_img * 255

    return masks_dilated

def compute_flow(frames, video_length, fix_raft):
    raft_iter = 20

    with torch.no_grad():
        if frames.size(-1) <= 640: 
            short_clip_len = 12
        elif frames.size(-1) <= 720: 
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
        
        # use fp32 for RAFT
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=raft_iter)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=raft_iter)
    
    return gt_flows_bi

def complete_flow(gt_flows_bi, subvideo_length, flow_masks, fix_flow_complete):
    with torch.no_grad():
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    flow_masks[:, s_f:e_f+1])
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    pred_flows_bi_sub, 
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
    
    return pred_flows_bi

def image_propagation(
        frames,
        masks_dilated,
        subvideo_length,
        video_length,
        pred_flows_bi,
        model,
        h,
        w
):
    with torch.no_grad():
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, subvideo_length) # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                       pred_flows_bi_sub, 
                                                                       masks_dilated[:, s_f:e_f], 
                                                                       'nearest')
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
        
    return updated_frames, updated_masks

def features_propagation_transformer(
        video_length, 
        subvideo_length, 
        h,
        w,
        updated_frames,
        masks_dilated,
        updated_masks,
        pred_flows_bi,
        model,
        frames_inp):
    ref_stride = 10
    neighbor_length = 10

    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = neighbor_length // 2
    if video_length > subvideo_length:
        ref_num = subvideo_length // ref_stride
    else:
        ref_num = -1

    for f in tqdm.tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
        
        with torch.no_grad():
            # 1.0 indicates mask
            l_t = len(neighbor_ids)
            
            # pred_img = selected_imgs # results of image propagation
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            
            pred_img = pred_img.view(-1, 3, h, w)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else: 
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
    
    return comp_frames

def inpaint(input_path, fix_raft, fix_flow_complete, model, use_half = True):
    subvideo_length = 80
    video = cv2.VideoCapture(input_path)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frames_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_len = 50

    frames = read_frame_from_videos(frame_gen_from_video(video))[:50]
    masks_dilated = infer_mask(frames)

    frames_tensor = to_tensor(frames).unsqueeze(0) * 2 - 1
    masks_dilated_tensor = to_tensor(masks_dilated).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_tensor, masks_dilated_tensor = frames_tensor.to(device), masks_dilated_tensor.to(device)
    print("frames_tensor.shape", frames_tensor.shape)

    gt_flows_bi = compute_flow(frames_tensor, frames_len, fix_raft)
    print("gt_flows_bi[0].shape", gt_flows_bi[0].shape)
    print("gt_flows_bi[1].shape", gt_flows_bi[1].shape)

    if use_half:
        frames_tensor, masks_dilated_tensor = frames_tensor.half(), masks_dilated_tensor.half()
        gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
    
    pred_flows_bi = complete_flow(gt_flows_bi, subvideo_length, masks_dilated_tensor, fix_flow_complete)
    print("pred_flows_bi[0].shape", pred_flows_bi[0].shape)
    print("pred_flows_bi[1].shape", pred_flows_bi[1].shape)
    
    updated_frames, updated_masks = image_propagation(
        frames_tensor,
        masks_dilated_tensor,
        subvideo_length,
        frames_len,
        pred_flows_bi,
        model,
        h,
        w)
    print("updated_frames.shape", updated_frames.shape)
    print("updated_masks.shape", updated_masks.shape)
    
    comp_frames = features_propagation_transformer(
        frames_len, 
        subvideo_length, 
        h,
        w,
        updated_frames,
        masks_dilated_tensor,
        updated_masks,
        pred_flows_bi,
        model,
        frames)
    print("np.shape(comp_frames)", np.shape(comp_frames))
    
    video.release()

    return comp_frames, masks_dilated