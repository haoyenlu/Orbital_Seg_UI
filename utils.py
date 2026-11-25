import numpy as np
import torch
from tqdm import tqdm
import SimpleITK as sitk
import utils_sitk
from monai.inferers import sliding_window_inference


def preprocess(simg: sitk.Image, clip_range: list = None, normalize=False):
    img_arr = sitk.GetArrayFromImage(simg)
    img_arr = img_arr.astype(np.float32)
    img_arr = np.transpose(img_arr, axes=(2,1,0))
    if clip_range is not None:
        a_min, a_max = clip_range[0], clip_range[1]
        img_arr = np.clip(img_arr, a_min, a_max)

    if normalize:
        img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))

    return img_arr



def pytorch_sliding_window_inference(model, 
                                     simg: sitk.Image, 
                                     params: dict, 
                                     sw_batch_size: int, 
                                     overlap: float, 
                                     target_class: int) -> sitk.Image:
    """
    model: trained PyTorch model
    simg : sitk Image data
    target_spacing: the desired spacing for prediction
    clip_range    : the intensity clipping range
    roi_size: size of sub-volume for inference (tuple of 3 ints)
    sw_batch_size: batch size for sliding window inference
    overlap: overlap ratio between sub-volumes (float between 0 and 1)
    """
    # Convert numpy array to torch tensor and add batch and channel dimensions

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cuda Available: {torch.cuda.is_available()}, Using device: {device}")

    target_spacing = params.get('spacing', [1.0,1.0,1.0])
    clip_range     = params.get('clip_range', [-1024,1024])
    roi_size       = params.get('roi_size', [32,32,32])
    normalize      = params.get('normalize',True)

    original_simg = sitk.Image(simg) # make copy of the original image

    # resample to target spacing
    original_spacing = simg.GetSpacing()
    if not np.allclose(target_spacing, original_spacing):
        simg = utils_sitk.resample(simg, target_spacing)

    # reorient to identity orientation
    original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        original_simg.GetDirection()
    )
    simg = utils_sitk.reorient_to_identity(simg)

    img_arr    = preprocess(simg, clip_range, normalize=normalize)
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).float().to(device)
    model.to(device)
    with torch.no_grad():
        model.eval()
        pred_tensor = sliding_window_inference(
            inputs=img_tensor, 
            roi_size=roi_size, 
            sw_batch_size=sw_batch_size, 
            predictor=model, 
            overlap=overlap,
            mode='gaussian',
            sigma_scale=0.125,
            progress=True,
        )
    
    # Convert the output tensor back to numpy array and remove batch and channel dimensions
    pred = torch.argmax(pred_tensor, dim=1)  # class index per voxel
    pred = pred.squeeze().cpu().numpy()
    pred = np.transpose(pred, axes=(2,1,0))
    labels = params.get('labels')
    if labels:
        # multi-label mask: keep the configured label values, everything else is background
        pred = np.where(np.isin(pred, list(labels.keys())), pred, 0).astype(np.uint8)
    else:
        pred = (pred == target_class).astype(np.uint8)

    # convert to sitk Image
    mask   = sitk.GetImageFromArray(pred)
    mask.CopyInformation(simg) # copy meta data information to segmentation result
    # if target_spacing != original_spacing:
    #     mask = utils_sitk.resample(mask, original_spacing, sitk.sitkNearestNeighbor)

    mask = sitk.DICOMOrient(mask, original_orientation)

    mask   = sitk.Resample(
        mask,
        original_simg,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID()
    )
    mask.CopyInformation(original_simg)

    return mask






def vPrediction(model, simg: sitk.Image, params: dict):

    """
    model: trained Keras model
    img: full resolution testing volume
    vSize: size of sub-volume
    stride: the stride for sampling the testing sub-volume,
            if not specified, stride=vSize
    """


    target_spacing = params.get('spacing', [1.0,1.0,1.0])
    clip_range     = params.get('clip_range', [-1024,1024])
    roi_size       = params.get('roi_size', [32,32,32])
    normalize      = params.get('normalize',True)
    stride         = params.get('stride', None)
    target_class   = params.get('target_class', 1)

    if not stride: stride=roi_size
    vSize = roi_size

    original_simg = sitk.Image(simg) # make copy of the original image

    # process simg data
    original_spacing = simg.GetSpacing()
    if not np.allclose(target_spacing, original_spacing):
        simg = utils_sitk.resample(simg, target_spacing, sitk.sitkLinear)

    # reorient to identity orientation (same as the pytorch pipeline)
    simg = utils_sitk.reorient_to_identity(simg)

    img_arr      = preprocess(simg, clip_range, normalize)
    imgSize      = img_arr.shape
    pad = []
    for i in range(3):
        # smallest padding so that the windows tile the volume exactly
        if imgSize[i] <= vSize[i]:
            padding = vSize[i] - imgSize[i]
        else:
            padding = (stride[i] - (imgSize[i]-vSize[i]) % stride[i]) % stride[i]
        pad.append(padding)

    # pad with background intensity (air), not 0 which is soft tissue in HU
    padImg = np.full((imgSize[0]+pad[0], imgSize[1]+pad[1], imgSize[2]+pad[2]), img_arr.min(), dtype=np.float32)
    padImg[:imgSize[0], :imgSize[1], :imgSize[2]] = img_arr
    scoreMap = np.zeros(padImg.shape)
    overlapMap = np.zeros(padImg.shape)

    x_steps = np.ceil((imgSize[0]+pad[0]-vSize[0]+1)/ stride[0])
    y_steps = np.ceil((imgSize[1]+pad[1]-vSize[1]+1)/ stride[1])
    z_steps = np.ceil((imgSize[2]+pad[2]-vSize[2]+1)/ stride[2])
    total_steps = int(x_steps * y_steps * z_steps)

    with tqdm(total=total_steps) as pbar:
        # iterate over the entire padded volume
        for x in range(0, imgSize[0]+pad[0]-vSize[0]+1, stride[0]):
            for y in  range(0, imgSize[1]+pad[1]-vSize[1]+1, stride[1]):
                for z in range(0, imgSize[2]+pad[2]-vSize[2]+1, stride[2]):
                    vol = padImg[x:(x+vSize[0]),y:(y+vSize[1]),z:(z+vSize[2])]
                    vol = np.expand_dims(np.expand_dims(vol, axis=0), axis=-1) # convert to tensor
                    pred = model.predict(vol)
                    pred = np.argmax(pred, axis=-1)
                    pred = (pred == target_class).astype(np.uint8)
                    pred = pred[0]         
                    scoreMap[x:(x+vSize[0]),y:(y+vSize[1]),z:(z+vSize[2])] += pred
                    overlapMap[x:(x+vSize[0]),y:(y+vSize[1]),z:(z+vSize[2])] += np.ones(pred.shape)
                    pbar.update(1)


    scoreMap   = scoreMap[:imgSize[0], :imgSize[1], :imgSize[2]]
    overlapMap = overlapMap[:imgSize[0], :imgSize[1], :imgSize[2]]
    out = np.divide(scoreMap, overlapMap)
    out[np.where(out>=0.5)]=1
    out[np.where(out<0.5)]=0
    out = out.astype(np.uint8)
    out = np.transpose(out, (2,1,0)) # xyz -> zyx


    # convert to sitk Image
    mask   = sitk.GetImageFromArray(out)
    mask.CopyInformation(simg) # copy meta data information to segmentation result

    # resample back onto the original image grid (handles spacing and orientation)
    mask   = sitk.Resample(
        mask,
        original_simg,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID()
    )

    return mask

