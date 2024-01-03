import os
from typing import List

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers

# I. NPY vs. NPZ
# As we already read from the doc, the .npy format is:
#
#       the standard binary file format in NumPy for
#       persisting a single arbitrary NumPy array on disk.
#       The format is designed to be as simple as possible while achieving its limited goals.
#
# And .npz is only a
#
#       simple way to combine multiple arrays into a single file,
#       one can use ZipFile to contain multiple “.npy” files.
#       We recommend using the file extension “.npz” for these archives. (sources)
#
# So, .npz is just a ZipFile containing multiple “.npy” files.
# And this ZipFile can be either compressed (by using np.savez_compressed) or
# uncompressed (by using np.savez).


class nnUNetDataset(object):
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        """
        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are dictionaries containing the relevant information for that case.
        dataset[training_case] -> info
        Info has the following key:value pairs:
        - dataset[case_identifier]['properties']['data_file'] -> the full path to the npz file associated with the training case
        - dataset[case_identifier]['properties']['properties_file'] -> the pkl file containing the case properties

        In addition, if the total number of cases is < num_images_properties_loading_threshold we load all the pickle files
        (containing auxiliary information). This is done for small datasets so that we don't spend too much CPU time on
        reading pkl files on the fly during training. However, for large datasets storing all the aux info (which also
        contains locations of foreground voxels in the images) can cause too much RAM utilization. In that
        case is it better to load on the fly.

        If properties are loaded into the RAM, the info dicts each will have an additional entry:
        - dataset[case_identifier]['properties'] -> pkl file content

        IMPORTANT! THIS CLASS ITSELF IS READ-ONLY. YOU CANNOT ADD KEY:VALUE PAIRS WITH nnUNetDataset[key] = value
        USE THIS INSTEAD:
        nnUNetDataset.dataset[key] = value
        (not sure why you'd want to do that though. So don't do it)
        """

        # 生成的是格式内容如下的字典：
        # {'0000': {'data_file': '0000.npz的地址',
        #           'properties_file': '0000.pkl的地址'},
        #  '0001': {'data_file': '0001.npz的地址',
        #           'properties_file': '0001.pkl的地址'},
        #  '0002': ...
        super().__init__()
        # print('loading dataset')
        if case_identifiers is None:
            # 把所有的case(0000,0001,0002...)都收集起来放进list case_identifiers
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        self.dataset = {}
        for c in case_identifiers:
            # 0001:{}
            self.dataset[c] = {}
            # 0001:{'data_file': '0001.npz的地址'}
            self.dataset[c]['data_file'] = join(folder, f"{c}.npz")
            # 0001:{'data_file': '0001.npz的地址',
            #       'properties_file': '0001.pkl的地址'}
            self.dataset[c]['properties_file'] = join(folder, f"{c}.pkl")

            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.npz")

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                # question 在properties的value里重抄了一遍properties_file，why？
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])
        # question ：nnUNet_keep_files_open 是什么时候写入 os.environ.keys()的？
        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))
        # print(f'nnUNetDataset.keep_files_open: {self.keep_files_open}')

    def __getitem__(self, key):
        # 难怪从Debugger里看，ds[0001]只有'data_file'和'properties_file',但是从Consle输出就会多一个'properties'
        ret = {**self.dataset[key]}
        # 每用[index]取值一次，就检查有没有'properties'这个key，没有的话就在这儿加上
        # 所以Debugger没有Console有！
        if 'properties' not in ret.keys():
            # 0001.pkl,这个pickle文件就是properties_file，里面装的就是properties_file
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key):
        # the instance of this class(i.e. self) has a key-based access mechanism
        # self本身就是一个字典套字典
        # 取出0001，相当于ds[0001]
        entry = self[key]
        # 看ds[0001]有没有'open_data_file'这个key
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + ".npy"):
            # entry['data_file'][:-4] + ".npy"是0001.npy的路径，如果这个文件存在，就把它读出来
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')

            if self.keep_files_open:
                # 如果keep_files_open,就把未压缩的nparray存到ds[0001]里，对应的key就是'open_data_file'
                self.dataset[key]['open_data_file'] = data
                # print('saving open data file')
        else:
            # entry['data_file']只有data和seg两个key
            data = np.load(entry['data_file'])['data']

        # 完全一样的操作，对seg再来一次
        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
                # print('saving open seg file')
        else:
            seg = np.load(entry['data_file'])['seg']

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))

        # 在第一行self[key]的时候就已经有'properties'了
        return data, seg, entry['properties']


if __name__ == '__main__':
    # this is a mini test. Todo: We can move this to tests in the future (requires simulated dataset)

    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset003_Liver/3d_lowres'
    myfolder = r'C:\Git\DataSet\nnunet\preprocessed\Dataset011_quarter_ACDC\nnUNetPlans_3d_fullres'
    ds = nnUNetDataset(myfolder, num_images_properties_loading_threshold=0) # this should not load the properties!
    # this SHOULD HAVE the properties
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # amazing. I am the best.

    # this should have the properties
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=1000)
    # now rename the properties file so that it does not exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now we should still be able to access the properties because they have already been loaded
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # move file back
    shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

    # this should not have the properties
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0)
    # now rename the properties file so that it does not exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now this should crash
    try:
        ks = ds['liver_0'].keys()
        raise RuntimeError('we should not have come here')
    except FileNotFoundError:
        print('all good')
        # move file back
        shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))


        # {'0000': {'data_file': 'C:\\Git\\DataSet\\nnunet\\preprocessed\\Dataset011_quarter_ACDC\\nnUNetPlans_3d_fullres\\0000.npz',
        #           'properties_file': 'C:\\Git\\DataSet\\nnunet\\preprocessed\\Dataset011_quarter_ACDC\\nnUNetPlans_3d_fullres\\0000.pkl'},
        #  '0001': {'data_file': 'C:\\Git\\DataSet\\nnunet\\preprocessed\\Dataset011_quarter_ACDC\\nnUNetPlans_3d_fullres\\0001.npz',
        #           'properties_file': 'C:\\Git\\DataSet\\nnunet\\preprocessed\\Dataset011_quarter_ACDC\\nnUNetPlans_3d_fullres\\0001.pkl'}

