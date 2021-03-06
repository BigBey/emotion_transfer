import os
import numpy as np
from itertools import groupby
from tqdm import tqdm


def write_split_npy(out_dir, data_file, data, train_indices, val_indices):
    if data is None:
        return
    train_data = data[train_indices]
    val_data = data[val_indices]
    base_name = os.path.splitext(os.path.basename(data_file))[0]
    train_path = os.path.join(out_dir, base_name + '_train.npy')
    val_path = os.path.join(out_dir, base_name + '_val.npy')
    np.save(train_path, train_data)
    np.save(val_path, val_data)


def main(in_dir, out_dir=None, ratio=0.1):
    # Validation
    if not os.path.isdir(in_dir):
        raise RuntimeError('Input directory does not exist: ' + in_dir)
    out_dir = in_dir if out_dir is None else out_dir
    if not os.path.isdir(out_dir):
        raise RuntimeError('Output directory does not exist: ' + out_dir)

    # Load data
    img_list_file = os.path.join(in_dir, 'img_list.txt')
    landmarks_file = os.path.join(in_dir, 'landmarks.npy')
    bboxes_file = os.path.join(in_dir, 'bboxes.npy')
    eulers_file = os.path.join(in_dir, 'eulers.npy')
    landmarks_3d_file = os.path.join(in_dir, 'landmarks_3d.npy')
    seg_list_file = os.path.join(in_dir, 'seg_list.txt')
    with open(img_list_file, 'r') as f:
        img_list = np.array(f.read().splitlines())
    landmarks = np.load(landmarks_file)
    bboxes = np.load(bboxes_file)
    eulers = np.load(eulers_file)
    landmarks_3d = np.load(landmarks_3d_file)
    with open(seg_list_file, 'r') as f:
        seg_list = np.array(f.read().splitlines())

    # Filter images without segmentations
    img_list_names = [os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.splitext(os.path.basename(f))[0])
                      for f in img_list]
    seg_list_names = [os.path.join(os.path.split(os.path.split(f)[0])[1], os.path.splitext(os.path.basename(f))[0])
                      for f in seg_list]
    valid_indices = [i for i in range(len(img_list_names)) if img_list_names[i] in seg_list_names]
    img_list = img_list[valid_indices]
    landmarks = landmarks[valid_indices]
    bboxes = bboxes[valid_indices]
    eulers = eulers[valid_indices]
    landmarks_3d = landmarks_3d[valid_indices]
    img_list_names = np.array(img_list_names)[valid_indices]

    # Generate directory splits
    val_indices = np.random.choice(len(img_list), int(np.round(len(img_list) * ratio)), replace=False).astype(int)
    train_indices = np.setdiff1d(np.arange(len(img_list)), val_indices)
    train_indices.sort()
    val_indices.sort()

    # Output splits to file
    train_img_list = img_list[train_indices]
    val_img_list = img_list[val_indices]
    base_name = os.path.splitext(os.path.basename(img_list_file))[0]
    train_split_path = os.path.join(out_dir, base_name + '_train.txt')
    val_split_path = os.path.join(out_dir, base_name + '_val.txt')
    np.savetxt(train_split_path, train_img_list, fmt='%s')
    np.savetxt(val_split_path, val_img_list, fmt='%s')

    train_seg_list = seg_list[train_indices]
    val_seg_list = seg_list[val_indices]
    base_name = os.path.splitext(os.path.basename(seg_list_file))[0]
    train_split_path = os.path.join(out_dir, base_name + '_train.txt')
    val_split_path = os.path.join(out_dir, base_name + '_val.txt')
    np.savetxt(train_split_path, train_seg_list, fmt='%s')
    np.savetxt(val_split_path, val_seg_list, fmt='%s')

    write_split_npy(out_dir, landmarks_file, landmarks, train_indices, val_indices)
    write_split_npy(out_dir, bboxes_file, bboxes, train_indices, val_indices)
    write_split_npy(out_dir, eulers_file, eulers, train_indices, val_indices)
    write_split_npy(out_dir, landmarks_3d_file, landmarks_3d, train_indices, val_indices)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('produce_masks_train_val')
    parser.add_argument('input', help='dataset root directory')
    parser.add_argument('-o', '--output', default=None, help='output directory')
    parser.add_argument('-r', '--ratio', default=0.1, type=float, help='ratio of validation split')
    args = parser.parse_args()
    main(args.input, args.output, ratio=args.ratio)
