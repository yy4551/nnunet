from typing import Union, Tuple

from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class nnUNetDataLoaderBase(DataLoader):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False):
        # def __init__(self,
        #               data, batch_size, num_threads_in_multithreaded=1,
        #               seed_for_shuffle=None, return_incomplete=False, shuffle=True,
        #               infinite=False, sampling_probabilities=None)
        super().__init__(data, batch_size, 1,
                         None, True, False,
                         True, sampling_probabilities)
        # data = _data
        # self.data的类型必须是nnUNetDataset的字典实例
        assert isinstance(data, nnUNetDataset), 'nnUNetDataLoaderBase only supports dictionaries as data'
        self.indices = list(data.keys())

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # question : self.indices和self.list_of_keys有什么区别？
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple(label_manager.all_labels) # all_labels已经把需要ignore的class排除在外了
        self.has_ignore = label_manager.has_ignore_label
        # 如果没开probabilistic_oversampling，就用_oversample_last_XX_percent
        # 如果开了，就用_probabilistic_oversampling
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        # self.oversample_foreground_percent：要对百分之多少的case进行fg_oversampling
        # 1-self.oversample_foreground_percent：对其余的case保留原状
        # 以batch为单位，
        # round(self.batch_size * (1 - self.oversample_foreground_percent))是一个batch中
        # 不需要oversample的case的个数
        # 如果sample_idx小于这个数，小于号表达式为真，取反，不做oversample
        # 小于号表达式为假，取反为真，做oversample
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        # 呃，这就是一个随机的摇骰子
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        # load one case
        # 从nnunet_dataset类的实例data的化名_data里取出一个第一个拿出来看看，就知道shape了
        data, seg, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        # a = (1,2,3)
        # b = (1,2,a)  # b = (1,2,(1,2,3))
        # b = (1,2,*a)  # b = (1,2,1,2,3)
        # batch_size + channels + patch_size,这就是一次卷积输入的5d tensor啊
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape


    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            # 只有patch_size过大 data_shape过小的时候才需要pad
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        # 在lbs和ubs之间选bbox，需要留出0.5*patch_size,这样取patch的时候才不会取到case image的外面啊
        # lbs和ubs是bbox_lbs的下界和上界
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            # 没开forcefg也没有需要无视的label，直接在lbs和ubs之间随便选
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                # 没开forcefg但有需要ignore的class，就把在用的class存到selected_class
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # class_locations[selected_class] 是一个list，如果len=0，那就说明image里根本没有
                    # 等于这个label值的点
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print('Warning! No annotated pixels in image!')
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
            elif force_fg:
                # 如果开了forcefg，就一定要以属于某个label的点为中心，选bbox
                # class_locations是一个字典，key是label值，value是一个list，list里存的是所有等于这个label值的点的坐标
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'

                if overwrite_class is not None:
                    # overwrite_class得写在class_locations里面才行，我不知道怎么写进去，所以没有试验它是什么功能
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                # eligible_classes_or_regions = [0,1,2,...]
                # 所谓eligible就是 这个label在class_locations里面有，且存在取这个label的点的位置坐标
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present,
                # remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous.
                # Use a.any() or a.all()

                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    # 如果有tmp存在，就说明存在一个目标区域，是所有的label的组合，remove？
                    if len(eligible_classes_or_regions) > 1:
                        # If any True value is present in tmp (i.e., if there is at least one element in
                        # eligible_classes_or_regions equal to self.annotated_classes_key),
                        # it further checks if there is more than one element in eligible_classes_or_regions.
                        # If both conditions are true, it removes the first occurrence of self.annotated_classes_key
                        # from the list using pop.
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # 一个foreground都没有，只有background
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    # 如果没开overwrite_class，就在eligible_classes_or_regions里随机抽一个label
                    # 如果开了overwrite_class，就要选这个overwrite_class作为selected_class
                    # 从此处可以推测，overwrite_class是用来指定一个class label，必须以这个label的点为中心选patch也即bbox
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')
            # voxels_of_that_class.shape = (1000,4),是label=selected_class的所有点的坐标
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                # 从刚才随机抽的label里再随机抽一个点，这个点(selected_voxel)就是bbox的中心点：
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                # 从selected_voxel这个点，减去0.5patch_size,就是patch的下界
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        # 从bbox_lbs这个下界，加上整个patch_size，就是patch的上界
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

if __name__ == '__main__':
    import pickle

    # folder = r'C:\Git\DataSet\nnunet\preprocessed\Dataset011_quarter_ACDC\nnUNetPlans_3d_fullres'
    # ds = nnUNetDataset(folder)  # this should not load the properties!
    ds = pickle.load(open(r'C:\Git\NeuralNetwork\nnunet\nnunetv2\training\dataloading\data_from_nnunet_dataset.pkl', 'rb'))
    label_dict =  {
        "background": 0,
        "F1+F2+F3":[1, 2, 3],
        "F2+F3":[2, 3],
        "F1+F3":[3]
    }

    temp = np.zeros([1, 64, 64, 64])
    temp[0, 10:40, 10:40, 10:40] = 1
    temp[0, 15:35, 15:35, 15:35] = 2
    temp[0, 20:30, 20:30, 20:30] = 3

    l1 = np.argwhere(temp == 1)
    l2 = np.argwhere(temp == 2)
    l3 = np.argwhere(temp == 3)
    class_locations={1:l1,2:l2,3:l3}


    lm = LabelManager(label_dict,[1,2,3])
    dl = nnUNetDataLoaderBase(ds, 5, (16, 16, 16), (16, 16, 16),lm,0.33, None)
    a = next(dl)
    lb1,ub1 = dl.get_bbox(np.array([64,64,64]),True,class_locations)
    print(lb1,ub1)

# for x in range(100,300):
#     a, b = dl.get_bbox(np.array([256, 256, 256]), True, class_locations)
#     assert [b[i]-a[i] for i in [0,1,2]]==[16,16,16],"oh sweety you wrong"
# print("congrats you right")
# output:
# congrats you right
# 可见对3d而言，这个方法是在image中生成patch区域