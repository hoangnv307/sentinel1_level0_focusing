# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sentinel-1 Level-0 → Focused SLC (configurable selected acquisition chunk, Stripmap/S6)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Processing flow:

    `L0 → raw decoding → I/Q correction → range compression → Doppler centroid → effective velocity → azimuth FFT → SRC → RCMC → azimuth compression → SLC`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 - Imports and input file
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 - Imports
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    from time import perf_counter

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import sentinel1decoder

    from IPython.display import display

    notebook_started_at = perf_counter()
    pd.set_option("display.max_columns", None)
    plt.style.use("default")
    return (
        PROJECT_ROOT,
        Path,
        colors,
        display,
        notebook_started_at,
        np,
        pd,
        perf_counter,
        plt,
        sentinel1decoder,
    )


@app.cell
def _(Path):
    import sentinel1_processing.raw_data_correction as raw_data_correction

    raw_correction_source = Path(raw_data_correction.__file__).read_bytes()
    return raw_correction_source, raw_data_correction


@app.cell
def _(Path):
    import sentinel1_processing.azimuth_pre_processing as azimuth_pre_processing
    import sentinel1_processing.range_processing as range_processing
    import sentinel1_processing.s6_parameters as s6_parameters

    _roots = (
        Path(azimuth_pre_processing.__file__).parent,
        Path(range_processing.__file__).parent,
    )
    range_source = tuple(
        path.read_bytes()
        for root in _roots
        for path in sorted(root.rglob("*.py"))
    ) + (Path(s6_parameters.__file__).read_bytes(),)
    return (
        azimuth_pre_processing,
        range_processing,
        range_source,
        s6_parameters,
    )


@app.cell
def _(Path, s6_parameters):
    import sentinel1_processing.doppler_centroid_estimation as doppler_centroid_estimation

    doppler_source = (
        Path(doppler_centroid_estimation.__file__).read_bytes(),
        Path(s6_parameters.__file__).read_bytes(),
    )
    return doppler_centroid_estimation, doppler_source


@app.cell
def _():
    import sentinel1_processing.dce_plotting as dce_plotting

    return (dce_plotting,)


@app.cell
def _(Path):
    import sentinel1_processing.effective_velocity as effective_velocity

    effective_velocity_source = Path(effective_velocity.__file__).read_bytes()
    return effective_velocity, effective_velocity_source


@app.cell
def _(Path, s6_parameters):
    import sentinel1_processing.azimuth_processing as azimuth_processing

    _root = Path(azimuth_processing.__file__).parent
    azimuth_source = tuple(
        path.read_bytes() for path in sorted(_root.rglob("*.py"))
    ) + (Path(s6_parameters.__file__).read_bytes(),)
    return azimuth_processing, azimuth_source


@app.cell
def _(np):
    def process_chunk(
        l0file,
        chunk,
        raw_slant_range_times_s,
        *,
        raw_data_correction,
        azimuth_pre_processing,
        range_reference_function,
        range_sample_freq,
        pulse_start_frequency_hz,
        pulse_ramp_rate_hz_per_s,
        pulse_length_s,
        output="valid",
        range_time_shift_s=0.0,
        output_array=None,
    ):
        radar_data = l0file.get_acquisition_chunk_data(chunk)
        preview = np.abs(radar_data[::20, ::20])
        iq_bias = raw_data_correction.estimate_iq_bias(radar_data)
        compressed, range_times = azimuth_pre_processing.range.compression.compress(
            radar_data,
            raw_slant_range_times_s,
            sample_rate_hz=range_sample_freq,
            pulse_start_frequency_hz=pulse_start_frequency_hz,
            pulse_ramp_rate_hz_per_s=pulse_ramp_rate_hz_per_s,
            pulse_length_s=pulse_length_s,
            iq_bias=np.complex128(iq_bias),
            range_reference_function=range_reference_function,
            range_time_shift_s=range_time_shift_s,
            output=output,
            output_array=output_array,
        )
        # sentinel1decoder caches every decoded chunk inside ``l0file`` for the
        # whole session (l0file.py ``get_acquisition_chunk_data``).  On this
        # scene a decoded chunk is 3-5 GB in RAM; the compressed result is what
        # the pipeline needs afterwards, so drop the raw reference now to keep
        # peak memory bounded instead of pinning every chunk until the notebook
        # session ends.  ``radar_data`` is a local alias and is freed on return.
        _raw_cache = getattr(l0file, "_acquisition_chunk_data_dict", None)
        if _raw_cache is not None and _raw_cache.get(chunk) is not None:
            _raw_cache[chunk] = None
        return compressed, range_times, (
            float(iq_bias.real), float(iq_bias.imag)
        ), preview

    return (process_chunk,)


@app.cell
def _():
    from notebook_support.cache import (
        array_cache_matches,
        cache_fingerprint,
        invalidate_broken_array_cache,
        open_array,
        prune_old_entries,
        save_cache_fingerprint,
        save_array,
    )

    return (
        array_cache_matches,
        cache_fingerprint,
        invalidate_broken_array_cache,
        open_array,
        prune_old_entries,
        save_array,
        save_cache_fingerprint,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 - Input product
    """)
    return


@app.cell
def _(PROJECT_ROOT, mo, sentinel1decoder):
    CACHE_ROOT = str(PROJECT_ROOT / ".cache" / "sentinel1")
    filepath = PROJECT_ROOT / "data" / "sao_paulo"
    filename = "s1a-s6-raw-s-vv-20251226t214356-20251226t214427-062491-07d496.dat"

    inputfile = str(filepath / filename)
    _input = mo.watch.file(inputfile)
    _input_stat = _input.stat()
    input_identity = (
        str(_input.resolve()),
        _input_stat.st_size,
        _input_stat.st_mtime_ns,
    )
    with mo.persistent_cache(
        name="level0-metadata", save_path=CACHE_ROOT, pin_modules=True
    ):
        input_identity
        l0file = sentinel1decoder.Level0File(inputfile)
        l0file.ephemeris
    return CACHE_ROOT, input_identity, l0file


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 - Metadata and acquisition chunk
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 - Packet metadata
    """)
    return


@app.cell
def _(l0file):
    l0file.packet_metadata
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 - Orbit records
    """)
    return


@app.cell
def _(l0file):
    l0file.ephemeris
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 - Acquisition chunk
    """)
    return


@app.cell
def _(l0file):
    selected_chunk = 14

    selection = l0file.get_acquisition_chunk_metadata(selected_chunk)
    len_az_line = len(selection)
    raw_len_range_line = 2 * int(selection["Number of Quads"].iloc[0])
    selection
    return len_az_line, raw_len_range_line, selected_chunk, selection


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 - Raw signal decoding (DAD §9.1)

    Decode the selected acquisition chunk into the complex I/Q matrix used by the Level-1 processing chain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 - Decode I/Q data
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    TXPL,
    TXPRR,
    TXPSF,
    azimuth_pre_processing,
    input_identity,
    invalidate_broken_array_cache,
    l0file,
    len_az_line,
    mo,
    np,
    open_array,
    process_chunk,
    range_reference_function,
    range_sample_freq,
    range_source,
    raw_correction_source,
    raw_data_correction,
    raw_slant_range_time_vec_s_1,
    save_array,
    selected_chunk,
):
    range_cache_directory = f"{CACHE_ROOT}/range-compression-{selected_chunk}"
    range_cache_file = f"{range_cache_directory}/data.npy"
    invalidate_broken_array_cache(
        range_cache_directory,
        range_cache_file,
        expected_shape=(
            len_az_line,
            len(raw_slant_range_time_vec_s_1)
            - int(np.ceil(TXPL * range_sample_freq))
            + 1,
        ),
    )

    def _process_to_file(path):
        compressed, range_times, bias, preview = process_chunk(
            l0file,
            selected_chunk,
            raw_slant_range_time_vec_s_1,
            raw_data_correction=raw_data_correction,
            azimuth_pre_processing=azimuth_pre_processing,
            range_reference_function=range_reference_function,
            range_sample_freq=range_sample_freq,
            pulse_start_frequency_hz=TXPSF,
            pulse_ramp_rate_hz_per_s=TXPRR,
            pulse_length_s=TXPL,
        )
        save_array(path, compressed)
        return range_times, bias, preview

    with mo.persistent_cache(
        name=f"range-compression-{selected_chunk}",
        save_path=CACHE_ROOT,
        pin_modules=True,
    ):
        input_identity, range_source, raw_correction_source
        (
            slant_range_time_vec_s,
            iq_bias_components,
            radar_data_preview,
        ) = _process_to_file(range_cache_file)
    range_compressed = open_array(range_cache_file)
    return (
        iq_bias_components,
        radar_data_preview,
        range_cache_file,
        range_compressed,
        slant_range_time_vec_s,
    )


@app.cell
def _(len_az_line, raw_len_range_line):
    print("Raw data shape:", (len_az_line, raw_len_range_line))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 - Raw data view
    """)
    return


@app.cell
def _(colors, plt, radar_data_preview):
    plt.figure(figsize=(12, 12))
    plt.title("Sentinel-1 Raw I/Q Sensor Output")
    plt.imshow(
        radar_data_preview,
        origin="lower",
        norm=colors.LogNorm()
    )
    plt.xlabel("Down Range (samples)")
    plt.ylabel("Cross Range (samples)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 - Input I/Q correction (DAD §9.2.1)

    Estimate the complex DC bias from the raw echo data before matched filtering.
    """)
    return


@app.cell
def _(iq_bias_components, np):
    iq_bias = np.complex128(*iq_bias_components)
    return (iq_bias,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 - Sampling axes (DAD §9.9)

    Build the range and azimuth time bases used by the processing chain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 - Radar and acquisition constants
    """)
    return


@app.cell
def _(s6_parameters, selection, sentinel1decoder):
    c = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS
    wavelength_m = s6_parameters.RADAR_WAVELENGTH_M

    RGDEC = selection["Range Decimation"].iloc[0]
    PRI = selection["PRI"].iloc[0]
    rank = selection["Rank"].iloc[0]
    TXPSF = selection["Tx Pulse Start Frequency"].iloc[0]
    TXPRR = selection["Tx Ramp Rate"].iloc[0]
    TXPL = selection["Tx Pulse Length"].iloc[0]
    return PRI, RGDEC, TXPL, TXPRR, TXPSF, c, rank, wavelength_m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 - Sampling frequencies

    Calculate the range sampling frequency and azimuth PRF from the metadata.
    """)
    return


@app.cell
def _(PRI, RGDEC, sentinel1decoder):
    range_sample_freq = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
    range_sample_period = 1.0 / range_sample_freq
    az_sample_freq = 1.0 / PRI
    az_sample_period = PRI
    suppressed_data_time = 320.0 / (8.0 * sentinel1decoder.constants.F_REF)
    return (
        az_sample_freq,
        az_sample_period,
        range_sample_freq,
        range_sample_period,
        suppressed_data_time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 - Slant-range grid

    Build the fast-time, slant-range time, and slant-range distance axes.
    """)
    return


@app.cell
def _(
    PRI,
    c,
    np,
    range_sample_freq,
    rank,
    raw_len_range_line,
    selection,
    suppressed_data_time,
):
    range_start_time = selection["SWST"].iloc[0] + suppressed_data_time
    raw_fast_time_vec_s = (
        range_start_time + np.arange(raw_len_range_line) / range_sample_freq
    )
    raw_slant_range_time_vec_s = rank * PRI + raw_fast_time_vec_s
    raw_slant_range_vec_m = raw_slant_range_time_vec_s * c / 2.0
    return raw_slant_range_time_vec_s, raw_slant_range_vec_m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4 - Azimuth-time grid

    Combine packet coarse and fine times on the GPS time base.
    """)
    return


@app.cell
def _(selection):
    packet_azimuth_times_s = (
        selection["Coarse Time"].to_numpy(dtype=float)
        + selection["Fine Time"].to_numpy(dtype=float)
    )
    return (packet_azimuth_times_s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.5 - Sampling summary
    """)
    return


@app.cell
def _(
    az_sample_freq,
    display,
    len_az_line,
    packet_azimuth_times_s,
    pd,
    range_sample_freq,
    raw_len_range_line,
    raw_slant_range_vec_m,
):
    sampling_summary = pd.Series({
        "range_samples": raw_len_range_line,
        "range_sample_frequency_mhz": range_sample_freq / 1e6,
        "azimuth_lines": len_az_line,
        "prf_hz": az_sample_freq,
        "slant_range_near_km": raw_slant_range_vec_m[0] / 1e3,
        "slant_range_far_km": raw_slant_range_vec_m[-1] / 1e3,
        "azimuth_start_gps_s": packet_azimuth_times_s[0],
        "azimuth_stop_gps_s": packet_azimuth_times_s[-1],
    })
    display(sampling_summary.to_frame("value"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Time base:** packet and ephemeris timestamps are GPS seconds. UTC annotation timestamps require the GPS–UTC offset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 - Range Processing (DAD §6.1)

    Steps: Range Reference Function → range dependent gain correction → SWST bias correction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 - Range Reference Function (DAD §6.1.1)

    Calculate the matched filter in the range-frequency domain.
    """)
    return


@app.cell
def _(
    TXPL,
    TXPRR,
    TXPSF,
    np,
    range_processing,
    range_sample_freq,
    raw_len_range_line,
):
    _tx_replica_sample_count = int(np.ceil(TXPL * range_sample_freq))
    range_fft_length = 1 << int(np.ceil(np.log2(raw_len_range_line + _tx_replica_sample_count - 1)))
    range_reference_function = range_processing.reference_function.calculate(sample_rate_hz=range_sample_freq, pulse_start_frequency_hz=TXPSF, pulse_ramp_rate_hz_per_s=TXPRR, pulse_length_s=TXPL, fft_length=range_fft_length)
    print(_tx_replica_sample_count)
    print(range_fft_length)
    return (range_reference_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 - Range Dependent Gain Correction (DAD §6.1.2)

    `range_processing.dependent_gain.apply()` is available when the gain vector is supplied. This dataset does not include that vector.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 - SWST Bias Correction (DAD §6.1.3)
    """)
    return


@app.cell
def _(range_processing, raw_slant_range_time_vec_s, s6_parameters):
    SWST_BIAS_S = s6_parameters.SWST_BIAS_S
    raw_slant_range_time_vec_s_1 = range_processing.swst_bias.correct(raw_slant_range_time_vec_s, SWST_BIAS_S)
    return SWST_BIAS_S, raw_slant_range_time_vec_s_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 - Azimuth Pre-Processing (DAD §6.2)
    - Input: An azimuth block of range-compressed data in 2D time domain.
    - Output: Data in range-Doppler domain, which is the required input for the next stage, azimuth processing
    - Stripmap S6 steps: azimuth zero-padding → range compression → forward azimuth FFT.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 - Azimuth Zero-Padding (DAD §6.2.1)

    `azimuth_pre_processing.azimuth_zero_padding.apply()` pads each processing block to the FFT length.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 - Range Compression (DAD §6.2.2)

    `azimuth_pre_processing.range.compression` exposes `zero_pad()`, `forward_fft()`, `multiply_reference_function()`, `inverse_fft()`, and `extract_valid_samples()`. `compress()` runs them in order.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Valid Range Samples
    """)
    return


@app.cell
def _(c, iq_bias, pd, range_compressed, slant_range_time_vec_s):
    slant_range_vec_m = slant_range_time_vec_s * c / 2.0
    len_range_line = range_compressed.shape[1]
    pd.Series({'iq_bias': iq_bias, 'range_compressed_shape': range_compressed.shape, 'valid_range_samples': len_range_line})
    return len_range_line, slant_range_vec_m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Range-compressed data
    """)
    return


@app.cell
def _(colors, np, plt, range_compressed):
    plt.figure(figsize=(12, 12))
    plt.title("After Range Compression")
    plt.imshow(
        np.abs(range_compressed[::20, ::20]),
        origin="lower",
        norm=colors.LogNorm()
    )
    plt.xlabel("Down Range (samples)")
    plt.ylabel("Cross Range (samples)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 - Azimuth Forward FFT (DAD §6.2.3)

    `azimuth_pre_processing.azimuth_forward_fft.apply()` transforms each padded block to the range-Doppler domain during focusing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 - Doppler Centroid Estimation (DAD §5)

    Estimate Doppler centroid from range-compressed data. Annotation polynomials are used only for comparison.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.1 - Annotation records

    Read the Level-1 annotation polynomials and Fine DC points used only as validation references; they are not inputs to focusing.
    """)
    return


@app.cell
def _(PROJECT_ROOT, doppler_centroid_estimation, np):
    from xml.etree import ElementTree

    _annotation_root = ElementTree.parse(
        PROJECT_ROOT / "references" /
        "s1a-s6-slc-vv-20251226t214357-20251226t214426-062491-07d496-002.xml"
    ).getroot()
    _annotation_records = []
    for _estimate in _annotation_root.findall(
        "./dopplerCentroid/dcEstimateList/dcEstimate"
    ):
        _fine_dce = _estimate.findall("./fineDceList/fineDce")
        _annotation_records.append({
            "azimuthTime": _estimate.findtext("azimuthTime"),
            "t0": float(_estimate.findtext("t0")),
            "geometryDcPolynomial": np.fromstring(
                _estimate.findtext("geometryDcPolynomial"), sep=" "
            ),
            "dataDcPolynomial": np.fromstring(
                _estimate.findtext("dataDcPolynomial"), sep=" "
            ),
            "dataDcRmsError": float(_estimate.findtext("dataDcRmsError")),
            "fineStart": _estimate.findtext("fineDceAzimuthStartTime"),
            "fineStop": _estimate.findtext("fineDceAzimuthStopTime"),
            "fineDceRangeTimes": np.array([
                float(point.findtext("slantRangeTime")) for point in _fine_dce
            ]),
            "fineDceFrequencies": np.array([
                float(point.findtext("frequency")) for point in _fine_dce
            ]),
        })
    DOPPLER_CENTROID_ANNOTATIONS = (
        doppler_centroid_estimation.parse_annotation_records(_annotation_records)
    )
    return (DOPPLER_CENTROID_ANNOTATIONS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.2 - Input segments (DAD §5.6)

    Load the adjacent acquisition chunk because the middle Doppler-estimation interval crosses the chunk boundary.
    """)
    return


@app.cell
def _(l0file, selected_chunk):
    adjacent_chunk = selected_chunk - 1
    adjacent_metadata = l0file.get_acquisition_chunk_metadata(adjacent_chunk)
    return adjacent_chunk, adjacent_metadata


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Trục thời gian cho chunk kề
    """)
    return


@app.cell
def _(
    SWST_BIAS_S,
    adjacent_metadata,
    np,
    range_processing,
    range_sample_freq,
    suppressed_data_time,
):
    raw_range_count_adjacent = 2 * int(adjacent_metadata["Number of Quads"].iloc[0])
    range_start_time_adjacent = adjacent_metadata["SWST"].iloc[0] + suppressed_data_time
    raw_slant_range_time_adjacent = (
        adjacent_metadata["Rank"].iloc[0] * adjacent_metadata["PRI"].iloc[0]
        + range_start_time_adjacent
        + np.arange(raw_range_count_adjacent) / range_sample_freq
    )
    raw_slant_range_time_adjacent = range_processing.swst_bias.correct(
        raw_slant_range_time_adjacent, SWST_BIAS_S
    )
    packet_azimuth_times_adjacent = (
        adjacent_metadata["Coarse Time"].to_numpy(dtype=float)
        + adjacent_metadata["Fine Time"].to_numpy(dtype=float)
    )
    return packet_azimuth_times_adjacent, raw_slant_range_time_adjacent


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.3 - Prepare the adjacent segment

    Apply the same I/Q correction and range-compression settings before joining both segments on a common grid.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.4 - Doppler centroid estimation (DAD §§5.2–5.5)

    Với sản phẩm S6 này, range-compression `valid` được gắn time axis tại
    zero-lag của matched filter. Fine DC dùng trung bình pha ACCC để tránh một
    vài range sample công suất lớn chi phối cả block. Unwrap dùng coherence để
    giảm ảnh hưởng của nghiệm noisy theo DAD §5.3; polynomial dùng least squares
    không trọng số và iterative RMS clipping ở ngưỡng 2.5 RMS. Cấu hình này tái
    tạo đúng valid mask và `dataDcRmsError` của ba record annotation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Doppler centroid time interval
    """)
    return


@app.cell
def _(
    PRI,
    az_sample_freq,
    doppler_centroid_estimation,
    packet_azimuth_times_adjacent,
    packet_azimuth_times_s,
):
    scene_start_acq_s = min(
        packet_azimuth_times_s[0], packet_azimuth_times_adjacent[0]
    )
    scene_stop_acq_s = max(
        packet_azimuth_times_s[-1], packet_azimuth_times_adjacent[-1]
    ) + PRI
    doppler_centroid_estimator = doppler_centroid_estimation.Estimator.for_stripmap_s6(
        prf_hz=az_sample_freq
    )
    return doppler_centroid_estimator, scene_start_acq_s, scene_stop_acq_s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Estimate Doppler centroid
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    Path,
    TXPL,
    TXPRR,
    TXPSF,
    adjacent_chunk,
    array_cache_matches,
    azimuth_pre_processing,
    cache_fingerprint,
    doppler_centroid_estimation,
    doppler_centroid_estimator,
    doppler_source,
    input_identity,
    l0file,
    mo,
    np,
    open_array,
    packet_azimuth_times_adjacent,
    packet_azimuth_times_s,
    process_chunk,
    range_cache_file,
    range_reference_function,
    range_sample_freq,
    range_source,
    raw_correction_source,
    raw_data_correction,
    raw_slant_range_time_adjacent,
    raw_slant_range_time_vec_s_1,
    s6_parameters,
    save_cache_fingerprint,
    scene_start_acq_s,
    scene_stop_acq_s,
    selected_chunk,
    slant_range_time_vec_s,
):
    _common_range_start = min(
        raw_slant_range_time_vec_s_1[0], raw_slant_range_time_adjacent[0]
    )

    def _dce_grid(range_times, azimuth_times):
        output_length = len(range_times) - int(np.ceil(TXPL * range_sample_freq)) + 1
        shift_s = (
            round((range_times[0] - _common_range_start) * range_sample_freq)
            / range_sample_freq
            - (range_times[0] - _common_range_start)
        )
        return shift_s, (len(azimuth_times), output_length)

    def _process_dce_to_file(path, chunk, range_times, azimuth_times):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        shift_s, output_shape = _dce_grid(range_times, azimuth_times)

        output = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.complex64,
            shape=output_shape,
        )
        compressed, output_times, _, _ = process_chunk(
            l0file,
            chunk,
            range_times,
            raw_data_correction=raw_data_correction,
            azimuth_pre_processing=azimuth_pre_processing,
            range_reference_function=range_reference_function,
            range_sample_freq=range_sample_freq,
            pulse_start_frequency_hz=TXPSF,
            pulse_ramp_rate_hz_per_s=TXPRR,
            pulse_length_s=TXPL,
            range_time_shift_s=shift_s,
            output="valid",
            output_array=output,
        )
        compressed.flush()
        temporary.replace(destination)
        return output_times

    def _prepare_dce_range(chunk, range_times, azimuth_times):
        directory = f"{CACHE_ROOT}/dce-range-compression-{chunk}"
        path = f"{directory}/data.npy"
        shift_s, shape = _dce_grid(range_times, azimuth_times)
        fingerprint = cache_fingerprint(
            "dce-range-v1",
            input_identity,
            chunk,
            raw_correction_source,
            range_source,
        )
        if not array_cache_matches(directory, path, fingerprint, shape):
            output_times = _process_dce_to_file(
                path, chunk, range_times, azimuth_times
            )
            save_cache_fingerprint(directory, fingerprint)
            return path, output_times
        return path, np.asarray(range_times[:shape[1]]) + shift_s

    def _estimate():
        selected_shift_s, _ = _dce_grid(
            raw_slant_range_time_vec_s_1, packet_azimuth_times_s
        )
        if selected_shift_s == 0.0:
            range_file_selected = range_cache_file
            slant_range_time_selected = slant_range_time_vec_s
        else:
            range_file_selected, slant_range_time_selected = _prepare_dce_range(
                selected_chunk,
                raw_slant_range_time_vec_s_1,
                packet_azimuth_times_s,
            )
        range_file_adjacent, slant_range_time_adjacent = _prepare_dce_range(
            adjacent_chunk,
            raw_slant_range_time_adjacent,
            packet_azimuth_times_adjacent,
        )
        range_compressed_selected = open_array(range_file_selected)
        range_compressed_adjacent = open_array(range_file_adjacent)

        # Order segments chronologically by their first azimuth line so the shared
        # Doppler interval crosses the chunk boundary cleanly.
        if packet_azimuth_times_s[0] <= packet_azimuth_times_adjacent[0]:
            ordered = [
                ("selected", range_compressed_selected, slant_range_time_selected,
                 packet_azimuth_times_s),
                ("adjacent", range_compressed_adjacent, slant_range_time_adjacent,
                 packet_azimuth_times_adjacent),
            ]
        else:
            ordered = [
                ("adjacent", range_compressed_adjacent, slant_range_time_adjacent,
                 packet_azimuth_times_adjacent),
                ("selected", range_compressed_selected, slant_range_time_selected,
                 packet_azimuth_times_s),
            ]
        segments = [
            doppler_centroid_estimation.Segment(
                data,
                range_time,
                az_times,
                name=label,
            )
            for label, data, range_time, az_times in ordered
        ]
        estimates, prepared = doppler_centroid_estimator.estimate_segments(
            segments,
            # N_amb shortcut đã kiểm chứng scene S6 (xem DCE_AMBIGUITY_NUMBER).
            # Muốn L0->L1 độc lập: thay bằng geometry_dc_provider tính từ orbit.
            known_ambiguity_number=s6_parameters.DCE_AMBIGUITY_NUMBER,
            slice_start_times_s=[scene_start_acq_s],
            last_slice_stop_time_s=scene_stop_acq_s,
            product_start_time_s=scene_start_acq_s,
            product_stop_time_s=scene_stop_acq_s,
            zero_dop_minus_acq_time_s=0.0,
            return_prepared_scene=True,
        )
        return estimates, prepared.alignment_summary()

    with mo.persistent_cache(
        name="doppler-centroid", save_path=CACHE_ROOT, pin_modules=True
    ):
        input_identity, doppler_source, range_source, raw_correction_source
        (
            doppler_centroid_estimates,
            doppler_centroid_alignment_summary,
        ) = _estimate()
    return doppler_centroid_alignment_summary, doppler_centroid_estimates


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Segment alignment
    """)
    return


@app.cell
def _(
    display,
    doppler_centroid_alignment_summary,
    doppler_centroid_estimates,
    pd,
):
    alignment_summary = pd.DataFrame(doppler_centroid_alignment_summary)
    display(alignment_summary)
    print("Estimated Doppler centroid records:", len(doppler_centroid_estimates))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.5 - Doppler centroid for each processing block

    Return the fitted Doppler centroid at each block center. Annotation data is not used here.
    """)
    return


@app.cell
def _(
    doppler_centroid_estimates,
    doppler_centroid_estimator,
    packet_azimuth_times_s,
    slant_range_time_vec_s,
):
    def doppler_centroid_for_block(block_center_index):
        return doppler_centroid_estimator.evaluate_at_line(
            doppler_centroid_estimates,
            line_index=block_center_index,
            azimuth_times_s=packet_azimuth_times_s,
            slant_range_times_s=slant_range_time_vec_s,
        )

    return (doppler_centroid_for_block,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.6 - Polynomial comparison

    Compare estimated and annotation polynomials. Report time error, RMSE, coherence, and PRF ambiguity.
    """)
    return


@app.cell
def _(
    DOPPLER_CENTROID_ANNOTATIONS,
    az_sample_freq,
    doppler_centroid_estimates,
    doppler_centroid_estimation,
    slant_range_time_vec_s,
):
    doppler_centroid_comparisons = doppler_centroid_estimation.compare_with_annotations(
        DOPPLER_CENTROID_ANNOTATIONS,
        doppler_centroid_estimates,
        slant_range_time_vec_s,
        prf_hz=az_sample_freq,
    )
    return (doppler_centroid_comparisons,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Error table
    """)
    return


@app.cell
def _(display, doppler_centroid_comparisons, pd):
    comparison_rows = []
    for result in doppler_centroid_comparisons:
        comparison_rows.append({
            "record": result["record"],
            "annotation_coefficients": result["annotation_coefficients"],
            "estimated_coefficients": result["estimated_coefficients"],
            "azimuth_time_error_ms": result["azimuth_time_error_ms"],
            "bias_hz": result["bias_hz"],
            "rmse_hz": result["rmse_hz"],
            "max_abs_error_hz": result["max_abs_error_hz"],
            "integer_prf_adjustment_hz": result["integer_prf_adjustment_hz"],
            "fit_rms_hz": result["fit_rms_hz"],
            "mean_coherence": result["mean_coherence"],
            "absolute_ambiguity_resolved": result["absolute_ambiguity_resolved"],
        })

    doppler_centroid_accuracy = pd.DataFrame(comparison_rows).set_index("record")
    display(doppler_centroid_accuracy)
    return (doppler_centroid_accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### So sánh từng điểm Fine DC với annotation

    Ghép một-một các record theo azimuth time và các điểm theo slant-range time.
    `valid_for_estimated_fit` là mask của phép fit tự tính; annotation không lưu
    mask mà IPF đã dùng.
    """)
    return


@app.cell
def _(
    DOPPLER_CENTROID_ANNOTATIONS,
    display,
    doppler_centroid_estimates,
    np,
    pd,
):
    fine_dc_rows = []
    annotations_by_time = sorted(
        DOPPLER_CENTROID_ANNOTATIONS, key=lambda item: item["azimuth_s"]
    )
    estimates_by_time = sorted(
        doppler_centroid_estimates,
        key=lambda item: item.block.azimuth_time_s,
    )
    if len(annotations_by_time) != len(estimates_by_time):
        raise ValueError("Số record Fine DC của annotation và estimate phải bằng nhau.")

    for record_index, (annotation, estimated) in enumerate(
        zip(annotations_by_time, estimates_by_time), start=1
    ):
        annotation_times = annotation["fineDceRangeTimes"]
        annotation_hz = annotation["fineDceFrequencies"]
        estimated_times = estimated.range_times_s
        estimated_hz = estimated.fine_baseband_hz
        if not (
            annotation_times.shape
            == annotation_hz.shape
            == estimated_times.shape
            == estimated_hz.shape
        ):
            raise ValueError("Fine DC annotation và estimate phải có cùng số điểm.")

        estimated_point_indices = np.array([
            np.argmin(np.abs(estimated_times - annotation_time))
            for annotation_time in annotation_times
        ])
        if np.unique(estimated_point_indices).size != annotation_times.size:
            raise ValueError("Không thể ghép một-một các điểm Fine DC theo range-time.")

        annotation_fit_hz = np.polynomial.polynomial.polyval(
            annotation_times - annotation["t0"],
            annotation["dataDcPolynomial"],
        )
        estimated_fit_hz = estimated.evaluate(annotation_times)
        azimuth_time_error_ms = 1e3 * (
            estimated.block.azimuth_time_s - annotation["azimuth_s"]
        )

        for annotation_point, estimated_point in enumerate(
            estimated_point_indices
        ):
            fine_dc_rows.append({
                "record": record_index,
                "point": annotation_point + 1,
                "azimuth_time_error_ms": azimuth_time_error_ms,
                "annotation_range_time_us": annotation_times[annotation_point] * 1e6,
                "estimated_range_time_us": estimated_times[estimated_point] * 1e6,
                "range_time_error_us": (
                    estimated_times[estimated_point]
                    - annotation_times[annotation_point]
                ) * 1e6,
                "annotation_fine_dc_hz": annotation_hz[annotation_point],
                "estimated_fine_dc_hz": estimated_hz[estimated_point],
                "fine_dc_error_hz": (
                    estimated_hz[estimated_point] - annotation_hz[annotation_point]
                ),
                "valid_for_estimated_fit": estimated.valid_mask[estimated_point],
                "coherence": estimated.coherence[estimated_point],
                "annotation_fit_hz": annotation_fit_hz[annotation_point],
                "estimated_fit_hz": estimated_fit_hz[annotation_point],
                "fit_curve_error_hz": (
                    estimated_fit_hz[annotation_point]
                    - annotation_fit_hz[annotation_point]
                ),
            })

    fine_dc_point_comparison = pd.DataFrame(fine_dc_rows).set_index(
        ["record", "point"]
    )
    fine_dc_summary_rows = []
    for record_index, group in fine_dc_point_comparison.groupby(level="record"):
        fine_error = group["fine_dc_error_hz"].to_numpy()
        valid = group["valid_for_estimated_fit"].to_numpy(dtype=bool)
        valid_error = fine_error[valid]
        fit_error = group["fit_curve_error_hz"].to_numpy()
        fine_dc_summary_rows.append({
            "record": record_index,
            "azimuth_time_error_ms": group["azimuth_time_error_ms"].iloc[0],
            "max_abs_range_time_error_us": np.max(
                np.abs(group["range_time_error_us"])
            ),
            "fine_all_bias_hz": fine_error.mean(),
            "fine_all_rmse_hz": np.sqrt(np.mean(fine_error**2)),
            "fine_valid_rmse_hz": np.sqrt(np.mean(valid_error**2)),
            "fit_curve_rmse_hz": np.sqrt(np.mean(fit_error**2)),
            "rejected_points": np.count_nonzero(~valid),
        })
    fine_dc_point_summary = pd.DataFrame(fine_dc_summary_rows).set_index("record")
    display(fine_dc_point_summary)
    display(fine_dc_point_comparison)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Doppler centroid curves
    """)
    return


@app.cell
def _(dce_plotting, doppler_centroid_comparisons, plt):
    dce_figures = dce_plotting.plot_comparisons(doppler_centroid_comparisons)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Error summary
    """)
    return


@app.cell
def _(doppler_centroid_accuracy):
    print(
        "Mean RMSE:", doppler_centroid_accuracy["rmse_hz"].mean(), "Hz |",
        "Worst max abs error:", doppler_centroid_accuracy["max_abs_error_hz"].max(), "Hz",
    )
    print(
        "Absolute ambiguity resolved:",
        doppler_centroid_accuracy["absolute_ambiguity_resolved"].all(),
        "(False is expected without a geometry DC provider)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 - Effective velocity (DAD §9.10)

    Use the L0 orbit records to calculate effective velocity for each processing block. Time checks are handled by the Python API.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8.1 - Orbit model
    """)
    return


@app.cell
def _(effective_velocity, l0file, packet_azimuth_times_s, wavelength_m):
    effective_velocity_estimator = effective_velocity.Estimator.from_level0_product(
        l0file, wavelength_m
    )
    # Chunk 14 ends ~0.90 s after the last embedded state-vector epoch, so the
    # focus aperture needs a small extrapolation allowance (mirrors the 2-chunk
    # notebook). Chunk 13 lies fully inside the ephemeris window.
    effective_velocity_estimator.validate_time_coverage(
        packet_azimuth_times_s, max_extrapolation_s=1.0
    )
    return (effective_velocity_estimator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8.2 - Orbit time interval
    """)
    return


@app.cell
def _(effective_velocity_estimator, packet_azimuth_times_s, pd):
    pd.Series({
        "orbit_epochs": effective_velocity_estimator.orbit_times_s.size,
        "orbit_start_gps_s": effective_velocity_estimator.orbit_times_s[0],
        "orbit_stop_gps_s": effective_velocity_estimator.orbit_times_s[-1],
        "echo_start_gps_s": packet_azimuth_times_s[0],
        "echo_stop_gps_s": packet_azimuth_times_s[-1],
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8.3 - Effective velocity check

    Use the 1398 Hz Stripmap S6 azimuth bandwidth from AUX_PP1.
    """)
    return


@app.cell
def _(
    doppler_centroid_for_block,
    effective_velocity_estimator,
    len_az_line,
    packet_azimuth_times_s,
    s6_parameters,
    slant_range_vec_m,
):
    AZIMUTH_PROCESSING_BANDWIDTH_HZ = s6_parameters.FOCUS_AZIMUTH_BANDWIDTH_HZ
    velocity_check_index = len_az_line // 2
    velocity_check = effective_velocity_estimator.evaluate_block(
        block_center_time_s=packet_azimuth_times_s[velocity_check_index],
        slant_range_m=slant_range_vec_m,
        fdc_hz=doppler_centroid_for_block(velocity_check_index),
        azimuth_bandwidth_hz=AZIMUTH_PROCESSING_BANDWIDTH_HZ,
        n_control_points=9,
        range_polynomial_degree=2,
        return_diagnostics=True,
    )
    return AZIMUTH_PROCESSING_BANDWIDTH_HZ, velocity_check


@app.cell
def _(
    dce_plotting,
    len_range_line,
    pd,
    plt,
    slant_range_vec_m,
    velocity_check,
):
    effective_velocity_figure = dce_plotting.plot_effective_velocity(
        slant_range_vec_m, velocity_check
    )
    plt.show()

    pd.Series({
        "near_range_mps": velocity_check.vr_mps[0],
        "mid_range_mps": velocity_check.vr_mps[len_range_line // 2],
        "far_range_mps": velocity_check.vr_mps[-1],
        "max_fit_rms_m": max(
            point.fit_rms_m for point in velocity_check.control_points
        ),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 - Azimuth processing blocks (DAD §§9.12–9.13)

    Set the azimuth bandwidth and calculate the matched-filter support, overlap, and block step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.1 - Processing parameters
    """)
    return


@app.cell
def _(s6_parameters):
    FOCUS_FFT_LEN = s6_parameters.FOCUS_FFT_LENGTH
    EXTRA_AZIMUTH_PROCESSING_BLOCK_OVERLAP = (
        s6_parameters.EXTRA_AZIMUTH_OVERLAP_SAMPLES
    )
    return EXTRA_AZIMUTH_PROCESSING_BLOCK_OVERLAP, FOCUS_FFT_LEN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.2 - Block layout
    """)
    return


@app.cell
def _(
    AZIMUTH_PROCESSING_BANDWIDTH_HZ,
    EXTRA_AZIMUTH_PROCESSING_BLOCK_OVERLAP,
    FOCUS_FFT_LEN,
    az_sample_freq,
    azimuth_processing,
    doppler_centroid_for_block,
    effective_velocity_estimator,
    len_az_line,
    packet_azimuth_times_s,
    slant_range_vec_m,
    wavelength_m,
):
    azimuth_block_layout = azimuth_processing.processing_blocks.calculate_layout(
        len_az_line,
        slant_range_vec_m,
        packet_azimuth_times_s,
        doppler_centroid_for_block,
        effective_velocity_estimator,
        wavelength_m=wavelength_m,
        azimuth_sample_frequency_hz=az_sample_freq,
        processing_bandwidth_hz=AZIMUTH_PROCESSING_BANDWIDTH_HZ,
        fft_length=FOCUS_FFT_LEN,
        extra_overlap_samples=EXTRA_AZIMUTH_PROCESSING_BLOCK_OVERLAP,
    )
    output_geometry = azimuth_processing.processing_blocks.L1OutputGeometry.from_focus_support(
        packet_azimuth_times_s,
        len(slant_range_vec_m),
        azimuth_block_layout,
        azimuth_sample_period_s=1.0 / az_sample_freq,
    )
    return azimuth_block_layout, output_geometry


@app.cell
def _(azimuth_block_layout, pd):
    pd.Series({
        "support_samples": azimuth_block_layout.matched_filter_support_samples,
        "overlap_samples": azimuth_block_layout.overlap_samples,
        "step_samples": azimuth_block_layout.step_samples,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 - Azimuth Processing (DAD §6.3)

    Stripmap S6 steps: SRC → RCMC → azimuth compression. Range resampling (§6.3.3) is only used for TOPSAR.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 10.1 - Secondary Range Compression (DAD §6.3.1)

    `azimuth_processing.secondary_range_compression.apply()` applies SRC in the range-Doppler domain.
    """)
    return


@app.cell
def _(s6_parameters):
    SRC_SEGMENT_SAMPLES = s6_parameters.SRC_SEGMENT_SAMPLES
    return (SRC_SEGMENT_SAMPLES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 10.2 - Range Cell Migration Correction (DAD §6.3.2)

    `azimuth_processing.range_cell_migration_correction.apply()` uses sinc interpolation.
    """)
    return


@app.cell
def _(s6_parameters):
    RCMC_KERNEL_LENGTH = s6_parameters.RCMC_KERNEL_LENGTH
    RCMC_NUM_PHASES = s6_parameters.RCMC_PHASES
    return RCMC_KERNEL_LENGTH, RCMC_NUM_PHASES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 10.3 - Azimuth Compression (DAD §6.3.4)

    `azimuth_processing.azimuth_compression` exposes `calculate_matched_filter()`, `calculate_time_correction_filter()`, `apply_filters()`, and `inverse_fft()`. `compress()` runs them in order. The time correction is zero until fine bistatic and instrument timing values are supplied.
    """)
    return


@app.cell
def _(s6_parameters):
    AZIMUTH_TIME_CORRECTION_S = s6_parameters.AZIMUTH_TIME_CORRECTION_S
    return (AZIMUTH_TIME_CORRECTION_S,)


@app.cell
def _(
    AZIMUTH_PROCESSING_BANDWIDTH_HZ,
    AZIMUTH_TIME_CORRECTION_S,
    CACHE_ROOT,
    FOCUS_FFT_LEN,
    RCMC_KERNEL_LENGTH,
    RCMC_NUM_PHASES,
    SRC_SEGMENT_SAMPLES,
    array_cache_matches,
    az_sample_period,
    azimuth_block_layout,
    azimuth_processing,
    azimuth_source,
    c,
    cache_fingerprint,
    doppler_centroid_for_block,
    doppler_source,
    effective_velocity_estimator,
    effective_velocity_source,
    input_identity,
    open_array,
    output_geometry,
    packet_azimuth_times_s,
    range_cache_file,
    range_sample_freq,
    range_sample_period,
    range_source,
    raw_correction_source,
    save_array,
    save_cache_fingerprint,
    slant_range_vec_m,
    wavelength_m,
):
    focused_cache_directory = f"{CACHE_ROOT}/focused-slc"
    focused_cache_file = f"{focused_cache_directory}/data.npy"
    focused_cache_fingerprint = cache_fingerprint(
        "focus-selected-chunk-v1",
        input_identity,
        raw_correction_source,
        range_source,
        doppler_source,
        effective_velocity_source,
        azimuth_source,
    )

    def _focus_to_file(path):
        focused = azimuth_processing.processing_blocks.focus_slc(
            open_array(range_cache_file),
            slant_range_vec_m,
            packet_azimuth_times_s,
            doppler_centroid_for_block,
            effective_velocity_estimator,
            azimuth_block_layout,
            wavelength_m=wavelength_m,
            speed_of_light_mps=c,
            azimuth_sample_period_s=az_sample_period,
            range_sample_period_s=range_sample_period,
            range_sample_frequency_hz=range_sample_freq,
            processing_bandwidth_hz=AZIMUTH_PROCESSING_BANDWIDTH_HZ,
            fft_length=FOCUS_FFT_LEN,
            azimuth_time_correction_s=AZIMUTH_TIME_CORRECTION_S,
            src_segment_samples=SRC_SEGMENT_SAMPLES,
            rcmc_kernel_length=RCMC_KERNEL_LENGTH,
            rcmc_phases=RCMC_NUM_PHASES,
            output_geometry=output_geometry,
        )
        save_array(path, focused)

    if not array_cache_matches(
        focused_cache_directory,
        focused_cache_file,
        focused_cache_fingerprint,
        output_geometry.shape,
    ):
        _focus_to_file(focused_cache_file)
        save_cache_fingerprint(
            focused_cache_directory,
            focused_cache_fingerprint,
        )
    focused_image = open_array(focused_cache_file)
    return (focused_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 - Focused SLC output (DAD §9.12)

    Giữ các line đủ aperture trên zero-Doppler grid. Đây vẫn là internal SLC;
    post-processing Hamming, calibration và SAFE formatting chưa được áp dụng.
    """)
    return


@app.cell
def _(focused_image, pd):
    pd.Series({"slc_shape": focused_image.shape})
    return


@app.cell(hide_code=True)
def _(CACHE_ROOT, focused_image, prune_old_entries):
    focused_image
    removed_cache_entries = prune_old_entries(CACHE_ROOT)
    if removed_cache_entries:
        print(f"Đã xóa {removed_cache_entries} cache cũ.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12 - Display focused image
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 12.1 - Display scale
    """)
    return


@app.cell
def _(
    TXPL,
    np,
    range_sample_freq,
    raw_slant_range_time_vec_s_1,
    slant_range_time_vec_s,
):
    _tx_replica_sample_count = int(np.ceil(TXPL * range_sample_freq))
    display_amplitude_scale = np.sqrt(_tx_replica_sample_count)
    valid_range_start = int(np.rint((slant_range_time_vec_s[0] - raw_slant_range_time_vec_s_1[0]) * range_sample_freq))
    return display_amplitude_scale, valid_range_start


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 12.2 - Full SLC image
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 12.3 - Detail view
    """)
    return


@app.cell
def _(
    colors,
    display_amplitude_scale,
    focused_image,
    np,
    plt,
    valid_range_start,
):
    plt.figure(figsize=(12, 12), dpi=75)
    plt.title("Sentinel-1 Processed SAR Image - detail")
    plt.imshow(
        np.abs(focused_image[
            10000:11000,
            6500 - valid_range_start:8000 - valid_range_start,
        ]),
        origin="lower",
        norm=colors.LogNorm(
            vmin=300 / display_amplitude_scale,
            vmax=10000 / display_amplitude_scale,
        ),
    )
    plt.xlabel("Down Range (samples)")
    plt.ylabel("Cross Range (samples)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 12.4 - Runtime
    """)
    return


@app.cell
def _(notebook_started_at, perf_counter):
    elapsed = perf_counter() - notebook_started_at
    print(f"Total runtime: {elapsed:.2f} s ({elapsed / 60:.2f} min)")
    return


if __name__ == "__main__":
    app.run()
