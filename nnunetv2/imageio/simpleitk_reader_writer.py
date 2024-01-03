#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import SimpleITK as sitk


class SimpleITKIO(BaseReaderWriter):
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []
        for f in image_fnames:
            # 图片读进来
            itk_image = sitk.ReadImage(f)
            # 获取图片的spacing, origin, direction
            # 这些信息转成nparray后就没有了，所以要提前存下来
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            # itk转为nparray
            npy_image = sitk.GetArrayFromImage(itk_image)

            # 它在调个儿:
            # >>>import SimpleITK as sitk
            # >>>img = sitk.ReadImage(r"C:\\Git\\DataSet\\nnunet\\raw\\Dataset011_quarter_ACDC\\imagesTr\\0001_0000.nii.gz")
            # >>>img.GetSpacing()
            # (1.0, 1.0, 8.0)
            # >>>npy = sitk.GetArrayFromImage(img)
            # >>>npy.shape
            # (16, 256, 256)
            # 可见sitk这个库返回图像和返回spacing的顺序不一样！

            if npy_image.ndim == 2:
                # 2d
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif npy_image.ndim == 3:
                # 3d, as in original nnunet
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif npy_image.ndim == 4:
                # 4d, multiple modalities in one file
                spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
                pass
            else:
                raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNet_plot_dataset_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNet_plot_dataset_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        # images[0].shape : (1, 16, 256, 256)
        # stacked_images.shape : (1, 16, 256, 256)
        stacked_images = np.vstack(images)
        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict
        # 返回的是一个nparray图像和一个字典
        # dict = {'sitk_stuff': {'spacing':
        #                        'origin':
        #                        'direction':  },
        #         'spacing': }

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert seg.ndim == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        sitk.WriteImage(itk_image, output_fname, True)


if __name__ == "__main__":
    sitkrw = SimpleITKIO()
    img_path = [r"C:\Git\DataSet\nnunet\raw\Dataset011_quarter_ACDC\imagesTr\0000_0000.nii.gz",
                r"C:\Git\DataSet\nnunet\raw\Dataset011_quarter_ACDC\imagesTr\0001_0000.nii.gz",
                r"C:\Git\DataSet\nnunet\raw\Dataset011_quarter_ACDC\imagesTr\0002_0000.nii.gz",
                r"C:\Git\DataSet\nnunet\raw\Dataset011_quarter_ACDC\imagesTr\0003_0000.nii.gz",
                r"C:\Git\DataSet\nnunet\raw\Dataset011_quarter_ACDC\imagesTr\0004_0000.nii.gz",
                r"C:\Git\DataSet\nnunet\raw\Dataset011_quarter_ACDC\imagesTr\0005_0000.nii.gz"]

    seg_path = r"C:\Git\DataSet\nnunet\raw\Dataset011_quarter_ACDC\labelsTr\0001.nii.gz"
    img, img_dict = sitkrw.read_images(img_path)
    pass
