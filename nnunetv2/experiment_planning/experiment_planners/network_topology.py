from copy import deepcopy
import numpy as np
from log import logger

def get_shape_must_be_divisible_by(net_numpool_per_axis):
    logger.debug(f"given {net_numpool_per_axis},if you want to do this much times of pooling\n"
                 f"along each axis,then the shape of input\n"
                 f"must be divisible by 2**{net_numpool_per_axis}= {2 ** np.array(net_numpool_per_axis)}\n")
    return 2 ** np.array(net_numpool_per_axis)


def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisible by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp


def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):

    """
    this is the same as get_pool_and_conv_props_v2 from old nnunet

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :param max_numpool:
    :return:
    """
    # todo review this code
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))
    logger.debug(f"current_spacing: {current_spacing}, current_size: {current_size}")

    pool_op_kernel_sizes = [[1] * len(spacing)] # list of kernel size list,[[1,3,3],[1,3,3],[1,3,3]......]
    conv_kernel_sizes = [] # [1,3,3]

    num_pool_per_axis = [0] * dim # how many times of pooling each axis does
    kernel_size = [1] * dim # kernel size of pooling,like [1,2,2],...,[2,2,2],...
    logger.debug(f"create empty list for later use:"
                 f"pool_op_kernel_sizes: {pool_op_kernel_sizes}, conv_kernel_sizes: {conv_kernel_sizes},\n"
                 f"num_pool_per_axis: {num_pool_per_axis}, kernel_size: {kernel_size}\n")

    while True:
        logger.debug("create netrwork by name the size of kernel of each operation,and current size at every moment\n")
        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        logger.debug(f"an axis is valid for pooling if the size along is larger than {2*min_feature_map_size}\n")
        valid_axes_for_pool = [i for i in range(dim) if current_size[i] >= 2*min_feature_map_size]
        logger.debug(f"valid_axes_for_pool: {valid_axes_for_pool}")
        if len(valid_axes_for_pool) < 1:
            break

        spacings_of_axes = [current_spacing[i] for i in valid_axes_for_pool]

        # find axis that are within factor of 2 within smallest spacing
        min_spacing_of_valid = min(spacings_of_axes)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_spacing[i] / min_spacing_of_valid < 2]

        logger.debug("an axis is valid pooling if its spacing is not 2 times larger than the smallest spacing\n")
        logger.debug(f"valid_axes_for_pool: {valid_axes_for_pool}")

        # max_numpool constraint
        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes_for_pool) == 1:
            logger.debug("if there is only one axis valid for pooling,then we can not pool any more\n"
                         "unless it is larger than 3*min_feature_map_size\n")
            if current_size[valid_axes_for_pool[0]] >= 3 * min_feature_map_size:
                pass
            else:
                break
        if len(valid_axes_for_pool) < 1:
            break

        # now we need to find kernel sizes
        # kernel sizes are initialized to 1. They are successively set to 3 when their associated axis becomes within
        # factor 2 of min_spacing. Once they are 3 they remain 3

        for d in range(dim):
            if kernel_size[d] == 3:
                logger.debug(f"kernel_size[{d}] == 3,continue\n")
                continue
            else:
                if current_spacing[d] / min(current_spacing) < 2:
                    logger.debug(f"kernel_size[{d}] is smaller than 3,set to 3")
                    kernel_size[d] = 3

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]
        logger.debug(f"axes wont be pooled: {other_axes}")

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            logger.debug(f"working on dim{v}")
            pool_kernel_sizes[v] = 2
            logger.debug(f"pool_kernel_sizes: {pool_kernel_sizes}")
            num_pool_per_axis[v] += 1
            logger.debug(f"num_pool_per_axis: {num_pool_per_axis}")
            current_spacing[v] *= 2
            logger.debug(f"current_spacing: {current_spacing}")
            current_size[v] = np.ceil(current_size[v] / 2)
            logger.debug(f"current_size: {current_size}")
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1
            logger.debug(f"pool_kernel_sizes: {pool_kernel_sizes}")

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))
        logger.debug(f"pool_op_kernel_sizes: {pool_op_kernel_sizes}\n")
        logger.debug(f"conv_kernel_sizes: {conv_kernel_sizes}\n")
        #print(conv_kernel_sizes)

    logger.debug("call get_shape_must_be_divisible_by\n")
    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    logger.debug("call pad_shape,pad patch size to be divisible\n")
    patch_size = pad_shape(patch_size, must_be_divisible_by)


    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3]*dim)
    return num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, must_be_divisible_by
