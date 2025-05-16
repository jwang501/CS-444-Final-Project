import torch 
import numpy as np
import matplotlib as plt

def MatrixNMS(sorted_masks, sorted_scores, method='gauss', gauss_sigma=0.5):
    n = len(sorted_scores)
    sorted_masks = sorted_masks.reshape(n, -1)
    intersection = torch.mm(sorted_masks, sorted_masks.T)
    areas = sorted_masks.sum(dim=1).expand(n, n)
    union = areas + areas.T - intersection
    ious = (intersection / union).triu(diagonal=1)

    ious_cmax = ious.max(0)[0].expand(n, n).T
    if method == 'gauss':
        decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
    else:
        decay = (1 - ious) / (1 - ious_cmax)
    decay = decay.min(dim=0)[0]
    return sorted_scores * decay
    
def PostProcess(cate_pred_list, ins_pred_list, postprocess_cfg):
    n_levels = len(cate_pred_list)
    n_imgs = cate_pred_list[0].shape[0]
    n_channels = cate_pred_list[0].shape[-1]
    #featmap_size = seg_preds[0].size()[-2:]
    results = []
    for batch_idx in range(n_imgs):
      # 3872*3
      cat_pred_per_img = torch.cat([
          cate_pred_list[i][batch_idx].view(-1, n_channels).detach() for i in range(n_levels)
      ], dim = 0)
      # 3872 * H * W
      mask_pred_per_img = torch.cat([
          ins_pred_list[i][batch_idx].detach() for i in range(n_levels)
      ], dim = 0)

      result = PostProcessImg(cat_pred_per_img, mask_pred_per_img, postprocess_cfg)

      results.append(result)

    return results

def PostProcessImg(cat_pred_per_img, mask_pred_per_img, postprocess_cfg):
    # 3872*3 TRUE/FALSE
    indice = cat_pred_per_img > postprocess_cfg['cate_thresh']
    # Tensor, n element P > cate_thresh
    cat_pred_per_img = cat_pred_per_img[indice]

    # n (element P > cate_thresh) * 2 (class 0, 1, 2)
    indice = indice.nonzero()

    cat_labels = indice[:, 1]
    # n filtered mask_pred_per_img for P > cate_thresh
    mask_pred_per_img = mask_pred_per_img[indice[:, 0]]

    binary_mask = mask_pred_per_img > postprocess_cfg['mask_thresh']

    num_p = binary_mask.sum((1, 2)).float()

    maskness = (mask_pred_per_img * binary_mask).sum((1, 2)) / num_p

    cat_pred_per_img *= maskness

    # NMS process
    sorted_index = torch.argsort(cat_pred_per_img, descending=True)
    sorted_index = sorted_index if len(sorted_index) <= postprocess_cfg['pre_NMS_num'] else sorted_index[:postprocess_cfg['pre_NMS_num']]

    sorted_masks = binary_mask[sorted_index].float()
    sorted_cat_pred = cat_pred_per_img[sorted_index]
    sorted_mask_preds = mask_pred_per_img[sorted_index]
    cat_labels = cat_labels[sorted_index]

    cat_pred_per_img = MatrixNMS(sorted_masks, sorted_cat_pred)

    # Keep top k
    sorted_index = torch.argsort(cat_pred_per_img, descending=True)
    sorted_index = sorted_index[:postprocess_cfg['keep_instance']]
    sorted_mask_preds = sorted_mask_preds[sorted_index]
    cat_pred_per_img = cat_pred_per_img[sorted_index]
    cat_labels = cat_labels[sorted_index]
    filter_idx_again = cat_pred_per_img > postprocess_cfg['cate_thresh']
    cat_pred_per_img = cat_pred_per_img[filter_idx_again]
    sorted_mask_preds = sorted_mask_preds[filter_idx_again]
    cat_labels = cat_labels[filter_idx_again]

    binary_masks = sorted_mask_preds > postprocess_cfg['mask_thresh']

    return binary_masks, cat_pred_per_img, cat_labels


import os
def PlotInfer(images, results, output_filename):
    batch_len = len(results)
    figure, axes_grid = plt.subplots(1, batch_len, figsize=(8 * batch_len, 8))
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]

    # Define color mapping for different labels
    label_colors = {
        0: [1, 0, 0, 0.5],   # Red for label 0
        1: [0, 1, 0, 0.5],   # Green for label 1
        2: [0, 0, 1, 0.5]    # Blue for label 2
    }

    images_np = images.cpu().numpy()
    images_np = np.moveaxis(images_np, 1, -1)
    #images_np = (images_np * std_values) + mean_values
    images_np = np.clip(images_np, 0, 1)
    

    for idx in range(batch_len):
        single_image = images_np[idx]

        current_axis = axes_grid[idx]
        result_masks = results[idx][0].cpu()
        result_labels = results[idx][2]
        result_scores = results[idx][1]
        current_axis.imshow(single_image)

        for mask_idx, single_mask in enumerate(result_masks):
            # if result_scores[mask_idx] < 0.2:
            #     continue

            # Create an overlay for the mask using the color mapped from the label
            overlay_img = np.zeros((single_image.shape[0], single_image.shape[1], 4))
            current_label = result_labels[mask_idx].item()

            # Resize the mask
            scaled_mask = Func.interpolate(single_mask[None, None, :, :].float(), size=(800, 1088), mode='nearest').squeeze()

            # Get the color for the current label or use a default (e.g., red) if the label is unknown
            color = label_colors[current_label]
            overlay_img[scaled_mask != 0] = color

            current_axis.imshow(overlay_img)

    plt.tight_layout()
    output_directory = os.path.join('solo_output', output_filename)
    plt.savefig(output_directory, dpi=80, bbox_inches='tight')

