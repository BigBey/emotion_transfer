""" Image to video face reenactment. """

import os
import face_alignment
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms.functional as F
from fsgan.data.landmark_transforms import crop_img, scale_bbox, Resize, generate_heatmaps
import fsgan.data.landmark_transforms as landmark_transforms
import fsgan.utils.utils as utils
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.video_utils import extract_landmarks_bboxes_euler_from_video
from fsgan.models.hopenet import Hopenet


def process_image(fa, img, size=256):
    detected_faces = fa.face_detector.detect_from_image(img.copy())
    if len(detected_faces) != 1:
        return None, None

    preds = fa.get_landmarks(img, detected_faces)
    landmarks = preds[0]
    bbox = detected_faces[0][:4]

    # Convert bounding boxes format from [min, max] to [min, size]
    bbox[2:] = bbox[2:] - bbox[:2] + 1

    return landmarks, bbox

    # scaled_bbox = scale_bbox(bbox)
    # cropped_img, cropped_landmarks = crop_img(img, landmarks, scaled_bbox)
    # landmarks_resize = Resize(size)
    # cropped_img, cropped_landmarks, scaled_bbox = \
    #     landmarks_resize(Image.fromarray(cropped_img), cropped_landmarks, scaled_bbox)
    #
    # return np.array(cropped_img), cropped_landmarks


def process_cached_frame(frame, landmarks, bbox, size=128):
    scaled_bbox = scale_bbox(bbox)
    cropped_frame, cropped_landmarks = crop_img(frame, landmarks, scaled_bbox)
    landmarks_resize = Resize(size)
    cropped_frame, cropped_landmarks, scaled_bbox = \
        landmarks_resize(Image.fromarray(cropped_frame), cropped_landmarks, scaled_bbox)

    return np.array(cropped_frame), cropped_landmarks


def unnormalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2bgr(img_tensor):
    output_img = unnormalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img[:, :, ::-1] * 255).astype('uint8')

    return output_img


def prepare_generator_input(img, landmarks, sigma=2):
    landmarks = generate_heatmaps(img.shape[1], img.shape[0], landmarks, sigma=sigma)
    landmarks = torch.from_numpy(landmarks)
    img = F.normalize(F.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return img, landmarks

def get_emotion_path(emotion):
    emotions_path = '../emotions/'
    emotions = {
        'com' : 'comfortable',
        'hap' : 'happy',
        'ins' : 'inspirational',
        'joy' : 'joy',
        'lon' : 'lonely',
        'fun' : 'funny',
        'nos' : 'nostalgic',
        'pas' : 'passionate',
        'qui' : 'quiet',
        'rel' : 'relaxed',
        'rom' : 'romantic',
        'sad' : 'sadness',
        'sou' : 'soulful',
        'swe' : 'sweet',
        'ser' : 'serious',
        'ang' : 'anger',
        'war' : 'wary',
        'sur' : 'surprise',
        'fea' : 'fear'
    }
    if emotion in emotions:
        return emotions_path + emotions[emotion] + '/' + emotions[emotion] + '_man.jpg'
    else:
        return None


def main(source_path, target_path,
         arch='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
         model_path='../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
         pose_model_path='../weights/hopenet_robust_alpha1.pth',
         pil_transforms1=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)'),
         pil_transforms2=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)', 'landmark_transforms.LandmarksToHeatmaps'),
         tensor_transforms1=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         tensor_transforms2=('landmark_transforms.ToTensor()',
                             'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         output_path=None, crop_size=256, display=False):
    torch.set_grad_enabled(False)

    # Initialize models
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    device = torch.device('cpu')
    G = obj_factory(arch).to(device)
    checkpoint = torch.load(model_path)
    G.load_state_dict(checkpoint['state_dict'])
    G.train(False)

    # Initialize pose
    Gp = Hopenet().to(device)
    checkpoint = torch.load(pose_model_path)
    Gp.load_state_dict(checkpoint['state_dict'])
    Gp.train(False)

    # Initialize transformations
    pil_transforms1 = obj_factory(pil_transforms1) if pil_transforms1 is not None else []
    pil_transforms2 = obj_factory(pil_transforms2) if pil_transforms2 is not None else []
    tensor_transforms1 = obj_factory(tensor_transforms1) if tensor_transforms1 is not None else []
    tensor_transforms2 = obj_factory(tensor_transforms2) if tensor_transforms2 is not None else []
    img_transforms1 = landmark_transforms.ComposePyramids(pil_transforms1 + tensor_transforms1)
    img_transforms2 = landmark_transforms.ComposePyramids(pil_transforms2 + tensor_transforms2)

    # Process source image
    source_bgr = cv2.imread(source_path)
    source_rgb = source_bgr[:, :, ::-1]
    source_landmarks, source_bbox = process_image(fa, source_rgb, crop_size)
    if source_bbox is None:
        raise RuntimeError("Couldn't detect a face in source image: " + source_path)
    source_tensor, source_landmarks, source_bbox = img_transforms1(source_rgb, source_landmarks, source_bbox)
    source_cropped_bgr = tensor2bgr(source_tensor[0] if isinstance(source_tensor, list) else source_tensor)
    for i in range(len(source_tensor)):
        source_tensor[i] = source_tensor[i].to(device)

    # Process target image
    target_bgr = cv2.imread(target_path)
    target_rgb = target_bgr[:, :, ::-1]
    target_landmarks, target_bbox = process_image(fa, target_rgb, crop_size)
    if target_bbox is None:
        raise RuntimeError("Couldn't detect a face in target image: " + target_path)
    target_tensor, target_landmarks, target_bbox = img_transforms2(target_rgb, target_landmarks, target_bbox)

    input_tensor = []
    for j in range(len(source_tensor)):
        target_landmarks[j] = target_landmarks[j].to(device)
        input_tensor.append(torch.cat((source_tensor[j], target_landmarks[j]), dim=0).unsqueeze(0).to(device))
    out_img_tensor, out_seg_tensor = G(input_tensor)

    # Convert back to numpy images
    out_img_bgr = tensor2bgr(out_img_tensor)
    frame_cropped_bgr = tensor2bgr(target_tensor[0])

    # Render
    render_img = np.concatenate((source_cropped_bgr, out_img_bgr, frame_cropped_bgr), axis=1)

    # Output
    output_name = '_'.join([os.path.splitext(os.path.basename(source_path))[0],
                                 os.path.splitext(os.path.basename(target_path))[0]]) + '.jpg'
    output_path = os.path.join(output_path, output_name)
    cv2.imwrite(output_path, render_img)

if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('reenactment')
    parser.add_argument('source', metavar='IMAGE',
                        help='path to source image')
    parser.add_argument('-e', '--emotion', type=str, metavar='IMAGE',
                        help='emotion, which can be on source face')
    parser.add_argument('-a', '--arch',
                        default='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
                        help='model architecture object')
    parser.add_argument('-m', '--model', default='../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth', metavar='PATH',
                        help='path to face reenactment model')
    parser.add_argument('-pm', '--pose_model', default='../weights/hopenet_robust_alpha1.pth', metavar='PATH',
                        help='path to face pose model')
    parser.add_argument('-pt1', '--pil_transforms1', nargs='+', help='first PIL transforms',
                        default=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                                 'landmark_transforms.Pyramids(2)'))
    parser.add_argument('-pt2', '--pil_transforms2', nargs='+', help='second PIL transforms',
                        default=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                                 'landmark_transforms.Pyramids(2)', 'landmark_transforms.LandmarksToHeatmaps'))
    parser.add_argument('-tt1', '--tensor_transforms1', nargs='+', help='first tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-tt2', '--tensor_transforms2', nargs='+', help='second tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output video path')
    parser.add_argument('-cs', '--crop_size', default=256, type=int, metavar='N',
                        help='crop size of the images')
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')
    args = parser.parse_args()

    if get_emotion_path(args.emotion) == None:
        print('ERROR: This emotion is unavailable.')
    else:
        main(args.source, get_emotion_path(args.emotion), args.arch, args.model, args.pose_model, args.pil_transforms1, args.pil_transforms2,
             args.tensor_transforms1, args.tensor_transforms2, args.output, args.crop_size, args.display)
