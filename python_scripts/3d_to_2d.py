import os, sys, pdb
from petrel_client.client import Client
import json
import numpy as np
import io
import gzip
import nibabel as nib
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import traceback
from PIL import Image


def list_dirs_and_files(client, url):
    sub_dirs = [d for d in client.list(url) if d.endswith('/')]
    sub_files = [f for f in client.list(url) if not f.endswith('/')]
    return sub_dirs, sub_files


def load_image(client, url):
    img_bytes = client.get(url)
    img = Image.open(io.BytesIO(img_bytes))
    return img


def save_image(client, url, img):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format('JPEG'), quality=95)
    client.put(url, img_bytes.getvalue())


def load_niigz_from_ceph(client, url):
    ''' V2: Uni test failed on several cases.
        * Attention: the numpy array shape of nibabel format: [x, y, z]
                     the numpy array shape of sitk format: [z, y, x]
    '''
    def nibabel_affine_to_direction(affine):
        # Extract the direction cosine matrix from the affine matrix
        return affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    
    with io.BytesIO(client.get(url, update_cache=True)) as f:
        f = gzip.open(f)
        header = nib.Nifti1Header.from_fileobj(f)
        array_proxy = nib.arrayproxy.ArrayProxy(f, header)
        nib_data = array_proxy.get_unscaled()
    itk_data = sitk.GetImageFromArray(nib_data.transpose(2, 1, 0))
    nii_affine = header.get_best_affine()
    
    # Origin
    origin = nii_affine[:3, 3]
    origin[:-1] *= -1
    itk_data.SetOrigin(origin)
    
    # Direction
    affine = nibabel_affine_to_direction(nii_affine)
    affine[:-1] *= -1
    direction = affine.flatten()
    itk_data.SetDirection(direction)
    
    # Spacing
    zooms = header.get_zooms()
    spacing = [float(z) for z in zooms]
    itk_data.SetSpacing(spacing)
    
    itk_data.SetMetaData('intent_code', str(header['intent_code']))
    itk_data.SetMetaData('intent_name', header['intent_name'].tobytes().decode('utf-8').strip())
    itk_data.SetMetaData('qform_code', str(header['qform_code']))
    itk_data.SetMetaData('sform_code', str(header['sform_code']))

    return itk_data


def repixel_value(arr, is_seg=False):
    if not is_seg:
        min_val = arr.min()
        max_val = arr.max()
        new_arr = (arr - min_val) / (max_val - min_val + 1e-10) * 255.
        new_arr = new_arr.astype(np.uint8)
    else:
        new_arr = arr.astype(np.uint8)
    img = Image.fromarray(new_arr).convert('RGB')
    return img


def split_3d_to_2d_with_axis(client, img, save_root, img_identifier, axis, is_seg=False):
    for i in range(img.shape[axis]):
        save_url = os.path.join(save_root, '{}_{}.jpg'.format(img_identifier, i))
        if client.contains(save_url):
            continue
        if axis == 0:
            img_2d = repixel_value(img[i, :, :], is_seg)
        elif axis == 1:
            img_2d = repixel_value(img[:, i, :], is_seg)
        elif axis == 2:
            img_2d = repixel_value(img[:, :, i], is_seg)
        else:
            raise Exception('value error of axis: {}'.format(axis))
        
        save_image(client, save_url, img_2d)


def split_3d_to_2d(client, dataset_root, save_root, with_seg=True):
    def process(img_path, lab_path):
        itk_img = load_niigz_from_ceph(client, os.path.join(dataset_root, img_path))
        img = sitk.GetArrayFromImage(itk_img) # x, y, z
        
        itk_lab = load_niigz_from_ceph(client, os.path.join(dataset_root, lab_path))
        lab = sitk.GetArrayFromImage(itk_lab) # x, y, z
        
        img_identifier = img_path[:-7]
        lab_identifier = lab_path[:-7]
        split_3d_to_2d_with_axis(client, img, os.path.join(save_root, 'x'), img_identifier, 0, False)
        split_3d_to_2d_with_axis(client, lab, os.path.join(save_root, 'x'), lab_identifier, 0, True)
        split_3d_to_2d_with_axis(client, img, os.path.join(save_root, 'y'), img_identifier, 1, False)
        split_3d_to_2d_with_axis(client, lab, os.path.join(save_root, 'y'), lab_identifier, 1, True)
        split_3d_to_2d_with_axis(client, img, os.path.join(save_root, 'z'), img_identifier, 2, False)
        split_3d_to_2d_with_axis(client, lab, os.path.join(save_root, 'z'), lab_identifier, 2, True)
    
    print ('* Processing: {}'.format(dataset_root))
    dataset_json = json.loads(client.get(os.path.join(dataset_root, 'dataset_after_check.json'), update_cache=True))

    tr_list = dataset_json.get('training', [])
    for case in tr_list:
        if 'label' not in case:
            break
        else:
            img_path = case['image']
            lab_path = case['label']
            process(img_path, lab_path)

    val_list = dataset_json.get('validation', [])
    for case in val_list:
        if 'label' not in case:
            break
        else:
            img_path = case['image']
            lab_path = case['label']
            process(img_path, lab_path)


    ts_list = dataset_json.get('test', [])
    for case in ts_list:
        if 'label' not in case:
            break
        else:
            img_path = case['image']
            lab_path = case['label']
            process(img_path, lab_path)


def run(dataset_root, save_root, with_seg=True):
    try:
        split_3d_to_2d(dataset_root, save_root, with_seg)
        print ('* Complete task: {}'.format(dataset_root))
    except Exception as e:
        print("Error in {}: {}".format(dataset_root, e))
        # print(traceback.format_exc())
        with open(os.path.join('error_info.txt'), 'a+') as fid:
            fid.write("Error in {}: {}\n".format(dataset_root, e))
            fid.write(traceback.format_exc())
        # return None


if __name__ == '__main__':
    client = Client('~/petreloss.conf', enable_mc=True)
    seg_root = 's3://medical_preprocessed/3d/semantic_seg/'
    save_root = 's3://medical_preprocessed/2d/semantic_seg_3d/'
    with_seg = True
    modalities, _ = list_dirs_and_files(client, seg_root)
    assert len(_) == 0

    # for mod in modalities:
    #     dataset_list, _ = list_dirs_and_files(client, os.path.join(seg_root, mod))
    #     for dataset in dataset_list:
    #         dataset_root = os.path.join(seg_root, mod, dataset)
    #         dataset_save_root = os.path.join(save_root, mod, dataset)
    #         split_3d_to_2d(client, dataset_root, dataset_save_root, with_seg=with_seg)

    with ThreadPoolExecutor(max_workers=64) as executor:
        tasks = []
        for mod in modalities:
            dataset_list, _ = list_dirs_and_files(client, os.path.join(seg_root, mod))
            for dataset in dataset_list:
                dataset_root = os.path.join(seg_root, mod, dataset)
                dataset_save_root = os.path.join(save_root, mod, dataset)
                future = executor.submit(split_3d_to_2d, client, dataset_root, dataset_save_root, with_seg=with_seg)
                tasks.append(future)
        
        results = [future.result() for future in as_completed(tasks)]

    print("Results:", results)
