import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            # properties['class_locations']是字典
            # {1：所有1的坐标位置   shape=(n1,4),n1是case中1的个数
            #  2：所有2的坐标位置   shape=(n2,4),n2是case中2的个数 ...}
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            # 只有bbox_lbs[i]为负值时才会padding
            # 只有bbox_ubs[i] > shape[i],也即bbox上界超出image的范围时才会padding
            # 如果超了，就会导致patch的实际尺寸不足，padding
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


if __name__ == '__main__':
    import pickle

    # folder = r'C:\Git\DataSet\nnunet\preprocessed\Dataset011_quarter_ACDC\nnUNetPlans_3d_fullres'
    # ds = nnUNetDataset(folder)  # this should not load the properties!
    ds = pickle.load(open(r'C:\Git\NeuralNetwork\nnunet\nnunetv2\training\dataloading\data_from_nnunet_dataset.pkl', 'rb'))

    from nnunetv2.utilities.label_handling.label_handling import LabelManager

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
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16),lm,0.33, None)
    dl.generate_train_batch()


