"""PDU §3.2.5.12 Stage-3 range sample-count calculation."""


def stage3_rx_samples(
    swl_code,
    *,
    interpolation_factor,
    decimation_factor,
    filter_output_offset,
    remainder_corrections,
):
    """Calculate the number of complex samples stored in a Space Packet."""
    swl = int(swl_code)
    interpolation = int(interpolation_factor)
    decimation = int(decimation_factor)
    corrections = tuple(int(value) for value in remainder_corrections)
    if swl < 0 or interpolation < 1 or decimation < 1:
        raise ValueError("SWL and decimation parameters must be positive.")
    if len(corrections) != decimation:
        raise ValueError("remainder_corrections must contain one D value per remainder.")

    b = 2 * swl - int(filter_output_offset) - 17
    quotient, remainder = divmod(b, decimation)
    return 2 * (interpolation * quotient + corrections[remainder] + 1)


def rgdec9_stage3_rx_samples(swl_code):
    """Stage-3 sample count for the RGDEC 9 filter used by S6."""
    return stage3_rx_samples(
        swl_code,
        interpolation_factor=5,
        decimation_factor=16,
        filter_output_offset=97,
        remainder_corrections=(
            0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,
        ),
    )


__all__ = ["stage3_rx_samples", "rgdec9_stage3_rx_samples"]
