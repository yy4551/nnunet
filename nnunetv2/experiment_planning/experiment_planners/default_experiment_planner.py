import os.path
import shutil
from copy import deepcopy
from functools import lru_cache
from typing import List, Union, Tuple, Type

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape, compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    get_filenames_of_train_images_and_targets
from log import logger

class ExperimentPlanner(object):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        """
        overwrite_target_spacing only affects 3d_fullres! (but by extension 3d_lowres which starts with fullres may
        also be affected
        """

        self.dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.suppress_transpose = suppress_transpose
        self.raw_dataset_folder = join(nnUNet_raw, self.dataset_name)
        preprocessed_folder = join(nnUNet_preprocessed, self.dataset_name)
        self.dataset_json = load_json(join(self.raw_dataset_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.raw_dataset_folder, self.dataset_json)

        # load dataset fingerprint
        if not isfile(join(preprocessed_folder, 'dataset_fingerprint.json')):
            raise RuntimeError('Fingerprint missing for this dataset. Please run nnUNet_extract_dataset_fingerprint')

        self.dataset_fingerprint = load_json(join(preprocessed_folder, 'dataset_fingerprint.json'))

        self.anisotropy_threshold = ANISO_THRESHOLD

        self.UNet_base_num_features = 32
        self.UNet_class = PlainConvUNet
        # the following two numbers are really arbitrary and were set to reproduce nnU-Net v1's configurations as
        # much as possible
        self.UNet_reference_val_3d = 560000000  # 455600128  550000000
        self.UNet_reference_val_2d = 85000000  # 83252480
        self.UNet_reference_com_nfeatures = 32
        self.UNet_reference_val_corresp_GB = 8
        self.UNet_reference_val_corresp_bs_2d = 12
        self.UNet_reference_val_corresp_bs_3d = 2
        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_featuremap_min_edge_length = 4
        self.UNet_blocks_per_stage_encoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        self.UNet_blocks_per_stage_decoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        self.UNet_min_batch_size = 2
        self.UNet_max_features_2d = 512
        self.UNet_max_features_3d = 320

        self.lowres_creation_threshold = 0.25  # if the patch size of fullres is less than 25% of the voxels in the
        # median shape then we need a lowres config as well

        self.preprocessor_name = preprocessor_name
        self.plans_identifier = plans_name
        self.overwrite_target_spacing = overwrite_target_spacing
        assert overwrite_target_spacing is None or len(overwrite_target_spacing), 'if overwrite_target_spacing is ' \
                                                                                  'used then three floats must be ' \
                                                                                  'given (as list or tuple)'
        assert overwrite_target_spacing is None or all([isinstance(i, float) for i in overwrite_target_spacing]), \
            'if overwrite_target_spacing is used then three floats must be given (as list or tuple)'

        self.plans = None

    def determine_reader_writer(self):
        logger.debug("determine_reader_writer")
        example_image = self.dataset[self.dataset.keys().__iter__().__next__()]['images'][0]
        return determine_reader_writer_from_dataset_json(self.dataset_json, example_image)

    @staticmethod
    @lru_cache(maxsize=None)
    def static_estimate_VRAM_usage(patch_size: Tuple[int],
                                   n_stages: int,
                                   strides: Union[int, List[int], Tuple[int, ...]],
                                   UNet_class: Union[Type[PlainConvUNet], Type[ResidualEncoderUNet]],
                                   num_input_channels: int,
                                   features_per_stage: Tuple[int],
                                   blocks_per_stage_encoder: Union[int, Tuple[int]],
                                   blocks_per_stage_decoder: Union[int, Tuple[int]],
                                   num_labels: int):
        logger.debug("static_estimate_VRAM_usage")
        """
        Works for PlainConvUNet, ResidualEncoderUNet
        """
        logger.debug("2d or 3d?\n")
        dim = len(patch_size)
        conv_op = convert_dim_to_conv_op(dim)         #nn.Conv1d/2d/3d
        norm_op = get_matching_instancenorm(conv_op)  #nn.InstanceNorm1d/2d/3d
        logger.debug(f"dim: {dim}, conv_op: {conv_op}, norm_op: {norm_op}\n")

        logger.debug(f"create instance of {UNet_class}\n")
        net = UNet_class(num_input_channels, n_stages,
                         features_per_stage,
                         conv_op,
                         3,
                         strides,
                         blocks_per_stage_encoder,
                         num_labels,
                         blocks_per_stage_decoder,
                         norm_op=norm_op)
        return net.compute_conv_feature_map_size(patch_size)

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        logger.debug("in determine_resampling,assign 'resample_data_or_seg_to_shape' to resampling_data\n"
                     "and resampling_seg,along with resampling_data_kwargs and resampling_seg_kwargs,for later use\n")
        resampling_data = resample_data_or_seg_to_shape
        resampling_data_kwargs = {
            "is_seg": False,
            "order": 3,
            "order_z": 0,
            "force_separate_z": None,
        }
        resampling_seg = resample_data_or_seg_to_shape
        resampling_seg_kwargs = {
            "is_seg": True,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        logger.debug("in determine_segmentation_softmax_export_fn,assign resample_data_or_seg_to_shape\n"
                     "to resampling_fn along with resampling_fn_kwargs for later use\n")
        resampling_fn = resample_data_or_seg_to_shape
        resampling_fn_kwargs = {
            "is_seg": False,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_fn, resampling_fn_kwargs

    def determine_fullres_target_spacing(self) -> np.ndarray:
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        """
        logger.debug("determine the full-res target spacing,\n"
                     "isotropic means the spacing of all axes are close to each other\n"
                     "anisotropic means the spacing of 1 axis is significantly larger than the others\n")
        if self.overwrite_target_spacing is not None:
            return np.array(self.overwrite_target_spacing)

        spacings = self.dataset_fingerprint['spacings']
        sizes = self.dataset_fingerprint['shapes_after_crop']

        # precentile is along the axis so assume from small to large
        target = np.percentile(np.vstack(spacings), 50, 0)

        # todo sizes_after_resampling = [compute_new_shape(j, i, target) for i, j in zip(spacings, sizes)]

        target_size = np.percentile(np.vstack(sizes), 50, 0)
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (self.anisotropy_threshold * max(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target

    def determine_normalization_scheme_and_whether_mask_is_used_for_norm(self) -> Tuple[List[str], List[bool]]:
        if 'channel_names' not in self.dataset_json.keys():
            print('WARNING: "modalities" should be renamed to "channel_names" in dataset.json. This will be '
                  'enforced soon!')
        modalities = self.dataset_json['channel_names'] if 'channel_names' in self.dataset_json.keys() else \
            self.dataset_json['modality']

        logger.debug(f"from json file we know modality is {modalities}")

        normalization_schemes = [get_normalization_scheme(m) for m in modalities.values()]   # it's object in list
        logger.debug(f"according to modality,search a mapping dict to decide normalization_schemes\n")

        if self.dataset_fingerprint['median_relative_size_after_cropping'] < (3 / 4.):
            logger.debug("median_relative_size_after_cropping in jason file <0.75,use mask")
            use_nonzero_mask_for_norm = [i.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true for i in
                                         normalization_schemes]
            logger.debug(f'there is a TRUE/FALSE flag in every normalization_scheme,extract it from each normalization_scheme\n'
                         f'maken them a list according which we do the mask\n')
        else:
            logger.debug("cool,we dont do mask \n")
            use_nonzero_mask_for_norm = [False] * len(normalization_schemes)
            assert all([i in (True, False) for i in use_nonzero_mask_for_norm]), 'use_nonzero_mask_for_norm must be ' \
                                                                                 'True or False and cannot be None'
        normalization_schemes = [i.__name__ for i in normalization_schemes] # it's string in list
        logger.debug(f"normalization_schemes = {normalization_schemes}\n"
                     f"also,return use_nonzero_mask_for_norm = {use_nonzero_mask_for_norm}")
        return normalization_schemes, use_nonzero_mask_for_norm

    def determine_transpose(self):
        logger.debug("transpose input volume to palce the axis with largest spacing on index 0,"
                     "so that the convolution is applied on the highest resolution plane as we expected\n")
        # what does this transpose for?
        if self.suppress_transpose:
            return [0, 1, 2], [0, 1, 2]

        # todo we should use shapes for that as well. Not quite sure how yet
        target_spacing = self.determine_fullres_target_spacing()

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]

        # we want the 2D model to operate on the highest resolution plane.
        # The 2D model always uses the last two dimensions of the 3D image (my_slice = image[slice_index, :, :]).
        # This is necessary due to the memory layout of the arrays.
        # image[:, slice_index, :] is MUCH slower
        transpose_forward = [max_spacing_axis] + remaining_axes

        # np.argwhere(np.array(transpose_forward) == i)[0][0] will return the indices where the value i is found in transpose_forward
        transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(3)]
        logger.debug(f"transpose_forward: {transpose_forward}, transpose_backward: {transpose_backward}")
        return transpose_forward, transpose_backward

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...], List[int]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float) -> dict:
        logger.debug("get_plans_for_configuration,a very long function\n")
        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        # print(spacing, median_shape, approximate_n_voxels_dataset)
        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
        # volume as a patch of size 256 ** 3)
        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
        # ideal because large initial patch sizes increase computation time because more iterations in the while loop
        # further down may be required.
        logger.debug("upscale a volume so it's certainly larger than what we need\n"
                     "namely, rescale to have the same volume as a patch of size 256 ** 3\n"
                     "np.prod(shape) = 256 ** 3\n")
        if len(spacing) == 3:
            initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            initial_patch_size = [round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError()

        # clip initial patch size to median_shape. It makes little sense to have it be larger than that. Note that
        # this is different from how nnU-Net v1 does it!
        # todo patch size can still get too large because we pad the patch size to a multiple of 2**n

        initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[:len(spacing)])])
        logger.debug("initial path size should not be larger than median_shape but acceptable if smaller than that,\n"
                     f"so take the minimum of calculated initial patch size and median_shape:{initial_patch_size}\n")

        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)
        logger.debug("\ncall get_pool_and_conv_props\n")
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                             self.UNet_featuremap_min_edge_length,
                                                             999999)

        # now estimate vram consumption
        num_stages = len(pool_op_kernel_sizes)
        logger.debug("1 pooling operation is 1 stage\n"
                     f"num_stages = {len(pool_op_kernel_sizes)}\n")

        logger.debug("\ncall static_estimate_VRAM_usage\n")
        estimate = self.static_estimate_VRAM_usage(tuple(patch_size),
                                                   num_stages,
                                                   tuple([tuple(i) for i in pool_op_kernel_sizes]),
                                                   self.UNet_class,
                                                   len(self.dataset_json['channel_names'].keys()
                                                       if 'channel_names' in self.dataset_json.keys()
                                                       else self.dataset_json['modality'].keys()),
                                                   tuple([min(self.UNet_max_features_2d if len(patch_size) == 2 else
                                                              self.UNet_max_features_3d,
                                                              self.UNet_reference_com_nfeatures * 2 ** i) for
                                                          i in range(len(pool_op_kernel_sizes))]),
                                                   self.UNet_blocks_per_stage_encoder[:num_stages],
                                                   self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                                                   len(self.dataset_json['labels'].keys()))

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        logger.debug(f"estimate is the output of static_estimate_VRAM_usage{estimate}, \n"
                      f"reference is given: {reference}\n")

        while estimate > reference:
            logger.debug("estimate > reference, we are going to reduce patch size\n")
            # print(patch_size)
            # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
            # aspect ratio the most (that is the largest relative to median shape)
            axis_to_be_reduced = np.argsort(patch_size / median_shape[:len(spacing)])[-1]
            logger.debug(f"take the aixs {axis_to_be_reduced} cuz it has the largest (patch_size / median_shape)\n")

            # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
            # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
            # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
            # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
            # subtract shape_must_be_divisible_by, then recompute it and then subtract the
            # recomputed shape_must_be_divisible_by. Annoying.
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, tmp,
                                        self.UNet_featuremap_min_edge_length,
                                        999999)
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            # now recompute topology
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                                 self.UNet_featuremap_min_edge_length,
                                                                 999999)

            num_stages = len(pool_op_kernel_sizes)
            estimate = self.static_estimate_VRAM_usage(tuple(patch_size),
                                                       num_stages,
                                                       tuple([tuple(i) for i in pool_op_kernel_sizes]),
                                                       self.UNet_class,
                                                       len(self.dataset_json['channel_names'].keys()
                                                           if 'channel_names' in self.dataset_json.keys()
                                                           else self.dataset_json['modality'].keys()),
                                                       tuple([min(self.UNet_max_features_2d if len(patch_size) == 2 else
                                                                  self.UNet_max_features_3d,
                                                                  self.UNet_reference_com_nfeatures * 2 ** i) for
                                                              i in range(len(pool_op_kernel_sizes))]),
                                                       self.UNet_blocks_per_stage_encoder[:num_stages],
                                                       self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                                                       len(self.dataset_json['labels'].keys()))

        # alright now let's determine the batch size. This will give self.UNet_min_batch_size if the while loop was
        # executed. If not, additional vram headroom is used to increase batch size
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        batch_size = round((reference / estimate) * ref_bs)
        logger.debug(f"batch_size is compputed as (reference / estimate) * ref_bs = {batch_size} ")

        # we need to cap the batch size to cover at most 5% of the entire dataset. (最高不超过dataset的5%)
        # Overfitting precaution. We cannot
        # go smaller than self.UNet_min_batch_size though
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * 0.05 / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)
        logger.debug(f"batch_szie can be no bigger than 5% of entire dataset,which is {bs_corresponding_to_5_percent}\n")

        logger.debug("\ncall determine_resampling\n")
        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        logger.debug("\ncall determine_segmentation_softmax_export_fn\n")
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        logger.debug("\ncall determine_normalization_scheme_and_whether_mask_is_used_for_norm\n")
        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
        num_stages = len(pool_op_kernel_sizes)
        logger.debug(f"num_stages = {num_stages},is this value changed? i don't think so\n")

        logger.debug("now we make up the whole plan using everything we got,and then we leave the get_plans_for_configuration\n")
        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'UNet_class_name': self.UNet_class.__name__,
            'UNet_base_num_features': self.UNet_base_num_features,
            'n_conv_per_stage_encoder': self.UNet_blocks_per_stage_encoder[:num_stages],
            'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
            'num_pool_per_axis': network_num_pool_per_axis,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
            'unet_max_num_features': self.UNet_max_features_3d if len(spacing) == 3 else self.UNet_max_features_2d,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
        }
        return plan

    def plan_experiment(self):
        """
        MOVE EVERYTHING INTO THE PLANS. MAXIMUM FLEXIBILITY

        Ideally I would like to move transpose_forward/backward into the configurations so that this can also be done
        differently for each configuration but this would cause problems with identifying the correct axes for 2d. There
        surely is a way around that but eh. I'm feeling lazy and featuritis must also not be pushed to the extremes.

        So for now if you want a different transpose_forward/backward you need to create a new planner. Also not too
        hard.
        """
        logger.debug("plan_experiment")

        # first get transpose
        logger.debug("call determine_transpose")
        transpose_forward, transpose_backward = self.determine_transpose()

        # get fullres spacing and transpose it
        logger.debug("call determine_fullres_target_spacing")
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        # get transposed new median shape (what we would have after resampling)
        new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(np.prod(new_median_shape_transposed, dtype=np.float64) *
                                             self.dataset_json['numTraining'])
        logger.debug(f"total voxel number of the whole dataset: {approximate_n_voxels_dataset}")

        # only run 3d if this is a 3d dataset
        if new_median_shape_transposed[0] != 1:
            logger.debug("call get_plans_for_configuration\n")
            plan_3d_fullres = self.get_plans_for_configuration(fullres_spacing_transposed,
                                                               new_median_shape_transposed,
                                                               self.generate_data_identifier('3d_fullres'),
                                                               approximate_n_voxels_dataset)
            logger.debug("plan_3d_fullres is the output of get_plans_for_configuration\n")

            # maybe add 3d_lowres as well
            patch_size_fullres = plan_3d_fullres['patch_size']
            median_num_voxels = np.prod(new_median_shape_transposed, dtype=np.float64)
            num_voxels_in_patch = np.prod(patch_size_fullres, dtype=np.float64)

            plan_3d_lowres = None
            lowres_spacing = deepcopy(plan_3d_fullres['spacing'])

            spacing_increase_factor = 1.03  # used to be 1.01 but that is slow with new GPU memory estimation!

            logger.debug("creating 3d low res plan\n")
            while num_voxels_in_patch / median_num_voxels < self.lowres_creation_threshold:
                # we incrementally increase the target spacing. We start with the anisotropic axis/axes until it/they
                # is/are similar (factor 2) to the other ax(i/e)s.
                max_spacing = max(lowres_spacing)
                if np.any((max_spacing / lowres_spacing) > 2):
                    logger.debug("the minimun spacing axis is 2 times smaller than the maximum spacing axis\n")
                    lowres_spacing[(max_spacing / lowres_spacing) > 2] *= spacing_increase_factor
                    logger.debug(f"incrementally enlarge,lowres_spacing = {lowres_spacing[(max_spacing / lowres_spacing) > 2]}\n")
                else:
                    lowres_spacing *= spacing_increase_factor
                    logger.debug(f"fullres spacing is quite isotropic,so enlarge them together,lowres_spacing={lowres_spacing}\n")

                median_num_voxels = np.prod(plan_3d_fullres['spacing'] / lowres_spacing * new_median_shape_transposed,
                                            dtype=np.float64)


                # print(lowres_spacing)
                logger.debug("creating plan_3d_lowres")
                plan_3d_lowres = self.get_plans_for_configuration(lowres_spacing,
                                                                  [round(i) for i in plan_3d_fullres['spacing'] /
                                                                   lowres_spacing * new_median_shape_transposed],
                                                                  self.generate_data_identifier('3d_lowres'),
                                                                  float(np.prod(median_num_voxels) *
                                                                        self.dataset_json['numTraining']))


                num_voxels_in_patch = np.prod(plan_3d_lowres['patch_size'], dtype=np.int64)
                print(f'Attempting to find 3d_lowres config. '
                      f'\nCurrent spacing: {lowres_spacing}. '
                      f'\nCurrent patch size: {plan_3d_lowres["patch_size"]}. '
                      f'\nCurrent median shape: {plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed}')

            if plan_3d_lowres is not None:
                plan_3d_lowres['batch_dice'] = False
                plan_3d_fullres['batch_dice'] = True
                logger.debug(f"low res plan exist,then low res batch_dice is false,full res batch_dice is true\n")
            else:
                plan_3d_fullres['batch_dice'] = False
                logger.debug("low res plan doesnt exist,full res batch_size is false")

        else:
            plan_3d_fullres = None
            plan_3d_lowres = None

        # 2D configuration
        plan_2d = self.get_plans_for_configuration(fullres_spacing_transposed[1:],
                                                   new_median_shape_transposed[1:],
                                                   self.generate_data_identifier('2d'), approximate_n_voxels_dataset)
        plan_2d['batch_dice'] = True

        print('2D U-Net configuration:')
        print(plan_2d)
        print()

        # median spacing and shape, just for reference when printing the plans
        median_spacing = np.median(self.dataset_fingerprint['spacings'], 0)[transpose_forward]
        median_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], 0)[transpose_forward]

        # instead of writing all that into the plans we just copy the original file. More files, but less crowded
        # per file.
        shutil.copy(join(self.raw_dataset_folder, 'dataset.json'),
                    join(nnUNet_preprocessed, self.dataset_name, 'dataset.json'))

        # json is stupid and I hate it... "Object of type int64 is not JSON serializable" -> my ass
        plans = {
            'dataset_name': self.dataset_name,
            'plans_name': self.plans_identifier,
            'original_median_spacing_after_transp': [float(i) for i in median_spacing],
            'original_median_shape_after_transp': [int(round(i)) for i in median_shape],
            'image_reader_writer': self.determine_reader_writer().__name__,
            'transpose_forward': [int(i) for i in transpose_forward],
            'transpose_backward': [int(i) for i in transpose_backward],
            'configurations': {'2d': plan_2d},
            'experiment_planner_used': self.__class__.__name__,
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': self.dataset_fingerprint[
                'foreground_intensity_properties_per_channel']
        }

        logger.debug("writing in plan-config details\n")
        if plan_3d_lowres is not None:
            plans['configurations']['3d_lowres'] = plan_3d_lowres
            if plan_3d_fullres is not None:
                plans['configurations']['3d_lowres']['next_stage'] = '3d_cascade_fullres'
            print('3D lowres U-Net configuration:')
            print(plan_3d_lowres)
            print()
        if plan_3d_fullres is not None:
            plans['configurations']['3d_fullres'] = plan_3d_fullres
            print('3D fullres U-Net configuration:')
            print(plan_3d_fullres)
            print()
            if plan_3d_lowres is not None:
                plans['configurations']['3d_cascade_fullres'] = {
                    'inherits_from': '3d_fullres',
                    'previous_stage': '3d_lowres'
                }

        self.plans = plans
        self.save_plans(plans)
        return plans

    def save_plans(self, plans):
        logger.debug("save_plans")
        recursive_fix_for_json_export(plans)

        plans_file = join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + '.json')

        # we don't want to overwrite potentially existing custom configurations every time this is executed. So let's
        # read the plans file if it already exists and keep any non-default configurations
        if isfile(plans_file):
            old_plans = load_json(plans_file)
            old_configurations = old_plans['configurations']
            for c in plans['configurations'].keys():
                if c in old_configurations.keys():
                    del (old_configurations[c])
            plans['configurations'].update(old_configurations)

        maybe_mkdir_p(join(nnUNet_preprocessed, self.dataset_name))
        save_json(plans, plans_file, sort_keys=False)
        print(f"Plans were saved to {join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + '.json')}")

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        logger.debug("\ngenerate_data_identifier\n")
        return self.plans_identifier + '_' + configuration_name

    def load_plans(self, fname: str):
        logger.debug("load_plans")
        self.plans = load_json(fname)


if __name__ == '__main__':
    ExperimentPlanner(11, 6).plan_experiment()
    # EPer = ExperimentPlanner(11, 6)
    # print(f"determine_reader_writer:{EPer.determine_reader_writer()}")
    # print(f"determine_resampling:{EPer.determine_resampling()}")