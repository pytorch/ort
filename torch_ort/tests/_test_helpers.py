# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch


def assert_values_are_close(input, other, rtol=1e-05, atol=1e-06):
    are_close = torch.allclose(input, other, rtol=rtol, atol=atol)
    if not are_close:
        abs_diff = torch.abs(input - other)
        abs_other = torch.abs(other)
        max_atol = torch.max((abs_diff - rtol * abs_other))
        max_rtol = torch.max((abs_diff - atol) / abs_other)
        err_msg = "The maximum atol is {}, maximum rtol is {}".format(max_atol, max_rtol)
        assert False, err_msg
