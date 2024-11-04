import sys
sys.path.append(".")

from mimo.configs.paths import PROPAINTER_REPO, CHECKPOINTS_FOLDER
sys.path.append(PROPAINTER_REPO)

import torch
import os, tqdm
import numpy as np
import scipy
from torch.utils.data import DataLoader

from mimo.utils.torch_utils import VideoDatasetSlidingWindow

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from inference_propainter import get_ref_index

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

class ProPainterBatchPredictor():
    def __init__(
        self, 
        batch_size: int,
        workers: int,
        use_half=True
    ):
        self.batch_size = batch_size
        self.workers = workers
        self.checkpoint_folder = CHECKPOINTS_FOLDER
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = use_half

        self.fix_raft, self.fix_flow_complete, self.model = self.load_models()

        self.subvideo_length = 80
        self.neighbor_stride = 5

    def collate(self, batch):
        # infer masks
        masks = self.infer_mask(batch[0])

        # the model expects RGB inputs
        batch = batch[0][:, :, :, ::-1] / 255.0 * 2 - 1
        batch = batch.astype("float32").transpose(0, 3, 1, 2)
        batch = torch.as_tensor(batch)
        return batch.unsqueeze(0), masks

    def inpaint(self, frame_gen):
        window_stride = self.batch_size - self.neighbor_stride*2
        dataset = VideoDatasetSlidingWindow(frame_gen, self.batch_size, window_stride)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.collate,
            pin_memory=True
        )
        with torch.no_grad():
            count_frames = 0
            for i_batch, (batch, masks_dilated) in enumerate(loader):
                batch_gpu = batch.to(self.device)
                masks_gpu = masks_dilated.to(self.device)

                flows_bi = self.compute_flow(batch_gpu)

                if self.use_half:
                    batch_gpu, masks_gpu = batch_gpu.half(), masks_gpu.half()
                    flows_bi = (flows_bi[0].half(), flows_bi[1].half())
                
                flows_bi = self.complete_flow(flows_bi, masks_gpu)

                update_batch_gpu, update_masks_gpu = self.image_propagation(
                    batch_gpu,
                    masks_gpu,
                    flows_bi)
                
                update_batch_gpu = self.features_propagation_transformer(
                    update_batch_gpu,
                    masks_gpu,
                    update_masks_gpu,
                    flows_bi,
                    batch_gpu)
            
                start = self.neighbor_stride
                end = -self.neighbor_stride
                if i_batch == 0:
                    start = 0
                if i_batch == len(loader)-1:
                    start = len(update_batch_gpu) - (dataset.num_frames-count_frames)
                    end = len(update_batch_gpu)
                
                output_images = update_batch_gpu[start:end]  # transformations already done
                output_mask = masks_gpu[0, start:end, 0, :, :].cpu().float().numpy()

                count_frames += len(output_images)

                yield output_images, output_mask

    def load_models(self):
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                        model_dir=self.checkpoint_folder, progress=True, file_name=None)
        fix_raft = RAFT_bi(ckpt_path, self.device)
        
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                        model_dir=self.checkpoint_folder, progress=True, file_name=None)
        fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
        for p in fix_flow_complete.parameters():
            p.requires_grad = False
        fix_flow_complete.to(self.device)
        fix_flow_complete.eval()

        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                        model_dir=self.checkpoint_folder, progress=True, file_name=None)
        model = InpaintGenerator(model_path=ckpt_path).to(self.device)
        model.eval()

        if self.use_half:
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        return fix_raft, fix_flow_complete, model

    @staticmethod
    def infer_mask(frames):
        mask_threshold = 0.1*255
        mask_dilates=4

        n_batch, w, h, _ = frames.shape
        masks_dilated = torch.zeros((1, n_batch, 1, w, h), dtype=torch.uint8)
            
        for i_frame in range(n_batch):
            mask_img = np.all(frames[i_frame, :, :, :] < mask_threshold, axis=-1).astype(np.uint8)
            
            if mask_dilates > 0:
                mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)

            masks_dilated[0, i_frame, 0, :, :] = torch.from_numpy(mask_img)

        return masks_dilated

    def compute_flow(self, frames):
        raft_iter = 20

        if frames.size(-1) <= 640: 
            short_clip_len = 12
        elif frames.size(-1) <= 720: 
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        _, n_batch, _, w, h = frames.shape
        
        # use fp32 for RAFT
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, n_batch, short_clip_len):
                end_f = min(n_batch, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = self.fix_raft(frames[:,f:end_f], iters=raft_iter)
                else:
                    flows_f, flows_b = self.fix_raft(frames[:,f-1:end_f], iters=raft_iter)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = self.fix_raft(frames, iters=raft_iter)
        
        return gt_flows_bi

    def complete_flow(self, gt_flows_bi, flow_masks):
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > self.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, self.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + self.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + self.subvideo_length)
                pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    flow_masks[:, s_f:e_f+1])
                pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    pred_flows_bi_sub, 
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
    
        return pred_flows_bi

    def image_propagation(
            self,
            frames,
            masks_dilated,
            pred_flows_bi,
    ):
        _, n_batch, _, w, h = frames.shape

        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, self.subvideo_length) # ensure a maximum of 100 frames for image propagation
        if n_batch > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, n_batch, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(n_batch, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(n_batch, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(masked_frames[:, s_f:e_f], 
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
            prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
        
        return updated_frames, updated_masks

    def features_propagation_transformer(
            self,
            updated_frames,
            masks_dilated,
            updated_masks,
            pred_flows_bi,
            frames_inp):
        ref_stride = 10
        
        _, n_batch, _, w, h = frames_inp.shape

        comp_frames = [None] * n_batch

        if n_batch > self.subvideo_length:
            ref_num = self.subvideo_length // ref_stride
        else:
            ref_num = -1

        for f in tqdm.tqdm(range(0, n_batch, self.neighbor_stride)):
            neighbor_ids = [
                i for i in range(max(0, f - self.neighbor_stride),
                                    min(n_batch, f + self.neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, n_batch, ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
            
            # 1.0 indicates mask
            l_t = len(neighbor_ids)
            
            # pred_img = selected_imgs # results of image propagation
            pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            
            pred_img = pred_img.view(-1, 3, h, w)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255

            ori_frames = (frames_inp[0] + 1) / 2
            ori_frames = ori_frames.cpu().permute(0, 2, 3, 1).numpy() * 255

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