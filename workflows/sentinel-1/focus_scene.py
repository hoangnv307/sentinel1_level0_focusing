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
    # Focus a complete Sentinel-1 Level-0 scene

    Luồng xử lý bám theo DAD Figure 5-1 và Figure 6-1 cho Stripmap/S6:

    1. đọc Level-0 và kiểm tra header (§4.3, §9.1);
    2. tạo RRF, sửa I/Q bias, range compression và black-fill (§6.1, §9.2.1,
       §6.2.2);
    3. ước lượng Doppler centroid từ dữ liệu đã range-compress (§5);
    4. xác định block focus Stripmap (§5.6, §9.10–§9.13);
    5. azimuth zero-padding/FFT, SRC, RCMC và azimuth compression
       (§6.2.1, §6.2.3, §6.3);
    6. lấy support SLC hợp lệ (§8.3.1).

    Demo không thực hiện các bước chỉ dành cho TOPSAR, Level-1
    post-processing/SAFE, drift, RFI, EAP hay range-spreading-loss correction.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    import matplotlib.pyplot as plt
    from matplotlib import colors
    import numpy as np
    import sentinel1decoder

    import notebook_support.cache as cache

    plt.style.use("default")
    return PROJECT_ROOT, cache, colors, np, plt, sentinel1decoder


@app.cell(hide_code=True)
def _(cache):
    import sentinel1_processing.azimuth_pre_processing as azimuth_pre_processing
    import sentinel1_processing.azimuth_processing as azimuth_processing
    import sentinel1_processing.common as common
    import sentinel1_processing.doppler_centroid as doppler_centroid
    import sentinel1_processing.pre_processing.downlink_header_validation as downlink_header_validation
    import sentinel1_processing.range_processing as range_processing
    import sentinel1_processing.s6_parameters as s6_parameters

    range_source = cache.source_snapshot(
        azimuth_pre_processing,
        downlink_header_validation,
        range_processing,
        s6_parameters,
    )
    raw_correction_source = cache.source_snapshot(common.raw_data_correction)
    doppler_source = cache.source_snapshot(doppler_centroid, s6_parameters)
    focus_source = cache.source_snapshot(
        azimuth_processing, common.effective_velocity, s6_parameters
    )
    return (
        azimuth_pre_processing,
        azimuth_processing,
        common,
        doppler_centroid,
        doppler_source,
        downlink_header_validation,
        focus_source,
        range_processing,
        range_source,
        raw_correction_source,
        s6_parameters,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 — Level-0 input và pre-processing metadata

    DAD §4.3, §9.1: mở Level-0, chọn các echo-data segment liên tiếp, kiểm tra
    downlink header và dựng trục thời gian range/azimuth từ metadata packet.
    """)
    return


@app.cell
def _(PROJECT_ROOT, cache, mo, sentinel1decoder):
    CACHE_ROOT = str(PROJECT_ROOT / ".cache" / "sentinel1")
    INPUT_PATH = (
        PROJECT_ROOT
        / "data"
        / "sao_paulo"
        / "s1a-s6-raw-s-vv-20251226t214356-20251226t214427-062491-07d496.dat"
    )
    input_identity = cache.file_identity(mo.watch.file(str(INPUT_PATH)))
    l0file = sentinel1decoder.Level0File(str(INPUT_PATH))
    return CACHE_ROOT, input_identity, l0file


@app.cell
def _(cache, l0file, s6_parameters, sentinel1decoder):
    # DAD §4.3: establish the downlink-header values required by later
    # processing stages. The scene spans two contiguous echo-data chunks in
    # this Level-0 product; downstream cells treat them as generic segments.
    SCENE_CHUNKS = (13, 14)
    SCENE_CACHE_KEY = cache.scene_cache_key(SCENE_CHUNKS)
    scene_metadata = tuple(
        l0file.get_acquisition_chunk_metadata(chunk) for chunk in SCENE_CHUNKS
    )

    # DAD §6 uses PRI for the azimuth sampling grid and Range Decimation for
    # the complex range sampling frequency. TXPSF, TXPRR, and TXPL describe
    # the transmitted chirp used to construct the Range Reference Function.
    c = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS
    wavelength_m = s6_parameters.RADAR_WAVELENGTH_M
    _first = scene_metadata[0]
    PRI = float(_first["PRI"].iloc[0])
    RGDEC = _first["Range Decimation"].iloc[0]
    TXPSF = float(_first["Tx Pulse Start Frequency"].iloc[0])
    TXPRR = float(_first["Tx Ramp Rate"].iloc[0])
    TXPL = float(_first["Tx Pulse Length"].iloc[0])
    range_sample_freq = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
    range_sample_period = 1.0 / range_sample_freq
    az_sample_freq = 1.0 / PRI
    suppressed_data_time = 320.0 / (8.0 * sentinel1decoder.constants.F_REF)
    return (
        PRI,
        SCENE_CACHE_KEY,
        SCENE_CHUNKS,
        TXPL,
        TXPRR,
        TXPSF,
        az_sample_freq,
        c,
        range_sample_freq,
        range_sample_period,
        scene_metadata,
        suppressed_data_time,
        wavelength_m,
    )


@app.cell
def _(
    PRI,
    downlink_header_validation,
    np,
    range_processing,
    range_sample_freq,
    s6_parameters,
    scene_metadata,
    sentinel1decoder,
    suppressed_data_time,
):
    SWST_BIAS_S = s6_parameters.SWST_BIAS_S

    def _axes(metadata):
        # DAD §4.3 / PDU §3.2.5.12: translate SWL and Range Decimation into
        # the expected Stage-3 complex sample count, then reject inconsistent
        # packets before any signal processing is attempted.
        _count = 2 * int(metadata["Number of Quads"].iloc[0])
        _swl_code = round(
            float(metadata["SWL"].iloc[0]) * sentinel1decoder.constants.F_REF
        )
        _expected = downlink_header_validation.sample_count.rgdec9_stage3_rx_samples(
            _swl_code
        )
        if _count != _expected:
            raise ValueError(
                f"PDU Stage-3 sample count is {_expected}, packet contains {_count}."
            )
        # DAD §6.2.2: Rank and SWST locate every decoded sample on the slant-
        # range time axis. DAD §6.1.3 then applies the instrument SWST bias.
        _raw_tau = (
            metadata["Rank"].iloc[0] * PRI
            + metadata["SWST"].iloc[0]
            + suppressed_data_time
            + np.arange(_count) / range_sample_freq
        )
        _tau = range_processing.swst_bias.correct(_raw_tau, SWST_BIAS_S)
        # Packet coarse and fine times form the slow-time/azimuth axis used by
        # DCE and all later azimuth-domain processing.
        _eta = (
            metadata["Coarse Time"].to_numpy(dtype=float)
            + metadata["Fine Time"].to_numpy(dtype=float)
        )
        return _tau, _eta, _count

    _axes_by_segment = tuple(_axes(metadata) for metadata in scene_metadata)
    raw_tau_segments, eta_segments, raw_range_counts = map(
        tuple, zip(*_axes_by_segment)
    )
    _common_start = min(tau[0] for tau in raw_tau_segments)

    def _fractional_shift(tau):
        # DAD §6.2.2.1: split an SWST displacement into integer and fractional
        # samples. The fractional part is applied as an RRF phase ramp during
        # compression; integer offsets are handled by black-fill alignment.
        _offset = (tau[0] - _common_start) * range_sample_freq
        return (round(_offset) - _offset) / range_sample_freq

    range_time_shifts = tuple(map(_fractional_shift, raw_tau_segments))
    return eta_segments, range_time_shifts, raw_range_counts, raw_tau_segments


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 — Range processing

    ### 2.1 — Range Reference Function (DAD §6.1.1)

    Tạo một RRF ở miền tần số với FFT đủ lớn cho SWL lớn nhất của scene.
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
    raw_range_counts,
):
    # DAD §6.1.1: build the normalized matched filter from the transmitted
    # chirp and transform it to range frequency. The FFT length covers linear
    # convolution for the largest SWL present in the complete scene.
    transmitted_pulse_samples = int(np.ceil(TXPL * range_sample_freq))
    _max_raw_samples = max(raw_range_counts)
    range_fft_length = 1 << int(np.ceil(np.log2(
        _max_raw_samples + transmitted_pulse_samples - 1
    )))
    range_reference_function = range_processing.range_reference_function.create_freq_domain(
        sample_rate_hz=range_sample_freq,
        pulse_start_frequency_hz=TXPSF,
        pulse_ramp_rate_hz_per_s=TXPRR,
        pulse_length_s=TXPL,
        fft_length=range_fft_length,
    )
    return range_reference_function, transmitted_pulse_samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 — Raw-data correction và range compression

    Với từng segment: giải mã BAQ (§9.1), ước lượng/sửa I/Q bias (§9.2.1), rồi
    zero-pad range, FFT, nhân RRF, IFFT và bỏ matched-filter transient (§6.2.2).
    """)
    return


@app.cell
def _(
    TXPL,
    TXPRR,
    TXPSF,
    azimuth_pre_processing,
    cache,
    common,
    l0file,
    np,
    range_reference_function,
    range_sample_freq,
    transmitted_pulse_samples,
):
    def compress_chunk(chunk, raw_tau, range_time_shift_s, destination):
        _radar_data = l0file.get_acquisition_chunk_data(chunk)
        # DAD §9.2.1: estimate and remove the constant I/Q bias immediately
        # after BAQ decoding. Other optional §9.2 corrections are not applied
        # by this demo.
        _iq_bias = np.complex128(
            common.raw_data_correction.estimate_iq_bias(_radar_data)
        )
        # DAD §6.2.2: SLC giữ Nraw - Ntx mẫu sau matched-filter throw-away.
        _shape = (
            _radar_data.shape[0],
            _radar_data.shape[1] - transmitted_pulse_samples,
        )

        def _write(output):
            # Trừ I/Q bias được gộp vào zero-padding để tránh tạo thêm một bản
            # sao toàn segment; thứ tự thuật toán vẫn là §9.2.1 trước §6.2.2.
            _, range_times = azimuth_pre_processing.range.compression.compress(
                _radar_data,
                raw_tau,
                sample_rate_hz=range_sample_freq,
                pulse_start_frequency_hz=TXPSF,
                pulse_ramp_rate_hz_per_s=TXPRR,
                pulse_length_s=TXPL,
                iq_bias=_iq_bias,
                range_reference_function=range_reference_function,
                range_time_shift_s=range_time_shift_s,
                output="slc",
                output_array=output,
            )
            return range_times

        range_times = cache.write_memmap(destination, _shape, _write)
        return range_times, (float(_iq_bias.real), float(_iq_bias.imag))

    return (compress_chunk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 — Materialize range-compressed segments

    Ghi từng segment đã range-compress vào memmap để các bước DCE và focus dùng
    chung mà không giữ toàn scene trong RAM.
    """)
    return


@app.cell(hide_code=True)
def _(
    CACHE_ROOT,
    SCENE_CACHE_KEY,
    SCENE_CHUNKS,
    cache,
    compress_chunk,
    eta_segments,
    input_identity,
    range_source,
    range_time_shifts,
    raw_correction_source,
    raw_tau_segments,
    transmitted_pulse_samples,
):
    def _load_chunk(chunk, eta, raw_tau, range_time_shift):
        shape = (len(eta), len(raw_tau) - transmitted_pulse_samples)
        fingerprint = cache.cache_fingerprint(
            "scene-range-v3", input_identity, SCENE_CACHE_KEY, chunk,
            range_time_shift, raw_correction_source, range_source,
        )
        path, _, created = cache.load_or_create_array(
            f"{CACHE_ROOT}/{SCENE_CACHE_KEY}/range-compression-{chunk}",
            fingerprint,
            shape,
            lambda destination: compress_chunk(
                chunk, raw_tau, range_time_shift, destination
            ),
        )
        if created is None:
            return (
                str(path),
                raw_tau[:shape[1]] + range_time_shift,
                (float("nan"),) * 2,
            )
        range_times, iq_bias = created
        return str(path), range_times, iq_bias

    _compressed_segments = tuple(
        _load_chunk(chunk, eta, raw_tau, range_time_shift)
        for chunk, eta, raw_tau, range_time_shift in zip(
            SCENE_CHUNKS,
            eta_segments,
            raw_tau_segments,
            range_time_shifts,
        )
    )
    range_cache_files, range_time_segments, iq_biases = map(
        tuple, zip(*_compressed_segments)
    )
    return iq_biases, range_cache_files, range_time_segments


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 — Black-fill và range-line length (DAD §6.2.2.1–§6.2.2.2)

    Phần lẻ của dịch chuyển SWST được sửa bằng phase ramp của RRF. Phần nguyên
    được đặt lên range grid chung bằng black-fill; dữ liệu phức không bị nội suy.
    Các segment sau đó được nối theo trục azimuth.
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    SCENE_CACHE_KEY,
    az_sample_freq,
    cache,
    doppler_centroid,
    doppler_source,
    eta_segments,
    input_identity,
    range_cache_files,
    range_time_segments,
):
    def make_segments():
        return [
            doppler_centroid.estimation.Segment(
                cache.open_array(path),
                range_times,
                azimuth_times,
                name=f"segment {index}",
            )
            for index, (path, range_times, azimuth_times) in enumerate(
                zip(range_cache_files, range_time_segments, eta_segments), start=1
            )
        ]

    def _combine_to_file(destination):
        # DAD §6.2.2.1-§6.2.2.2: SWST và SWL có thể đổi giữa các range line.
        # Map từng
        # segment onto one common slant-range grid, using black fill for the
        # integer offsets rather than resampling the complex measurements.
        _prepared = doppler_centroid.estimation.prepare_segments(
            make_segments(),
            prf_hz=az_sample_freq,
        )
        cache.write_memmap(
            destination,
            (_prepared.num_azimuth_lines, _prepared.num_range_samples),
            lambda output: _prepared.align_into(output, batch_lines=128),
        )
        return (
            _prepared.common_slant_range_times_s,
            _prepared.azimuth_times_s,
            _prepared.alignment_summary(),
        )

    with cache.persistent(
        "range-aligned", f"{CACHE_ROOT}/{SCENE_CACHE_KEY}"
    ):
        input_identity, doppler_source, range_cache_files
        combined_range_cache = (
            f"{CACHE_ROOT}/{SCENE_CACHE_KEY}/range-aligned/data.npy"
        )
        common_tau, combined_eta, alignment_summary = _combine_to_file(
            combined_range_cache
        )
    return (
        alignment_summary,
        combined_eta,
        combined_range_cache,
        common_tau,
        make_segments,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 — Doppler Centroid Estimation (DAD §5)

    Theo Figure 5-1: tính DC hình học từ orbit/attitude (§5.1), Fine DC bằng
    lag-one correlation (§5.2.2), unwrap (§5.3), giải ambiguity tuyệt đối
    (§5.4), rồi fit polynomial theo range và đo chất lượng (§5.5).
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    PRI,
    SCENE_CACHE_KEY,
    az_sample_freq,
    c,
    cache,
    doppler_centroid,
    doppler_source,
    input_identity,
    l0file,
    make_segments,
    raw_tau_segments,
    wavelength_m,
):
    doppler_estimator = doppler_centroid.estimation.Estimator.for_stripmap_s6(
        prf_hz=az_sample_freq
    )
    geometry_dc_estimator = doppler_centroid.geometry.Estimator.from_level0_product(
        l0file, wavelength_m
    )

    def _estimate_doppler():
        # DAD §5.2-§5.5: estimate fine DC with lag-one correlation, unwrap it,
        # resolve the PRF ambiguity against the §5.1 orbit/attitude geometry
        # estimate, then fit the range-dependent Doppler polynomials.
        _segments = make_segments()
        _start = float(_segments[0].azimuth_times_s[0])
        _stop = float(_segments[-1].azimuth_times_s[-1] + PRI)
        _estimates = doppler_estimator.estimate_segments(
            _segments,
            dce_range_start_s=float(raw_tau_segments[0][0]),
            geometry_dc_provider=lambda time_s, range_times_s: (
                geometry_dc_estimator.estimate(time_s, range_times_s * c / 2.0)
            ),
            slice_start_times_s=[_start],
            last_slice_stop_time_s=_stop,
            product_start_time_s=_start,
            product_stop_time_s=_stop,
            zero_dop_minus_acq_time_s=0.0,
        )
        return _estimates

    with cache.persistent(
        "doppler-centroid", f"{CACHE_ROOT}/{SCENE_CACHE_KEY}"
    ):
        input_identity, doppler_source
        doppler_estimates = _estimate_doppler()
    return doppler_estimates, doppler_estimator, geometry_dc_estimator


@app.cell
def _(combined_eta, common_tau, doppler_estimates, doppler_estimator):
    def doppler_centroid_for_line(
        line_index, slant_range_times_s=common_tau
    ):
        return doppler_estimator.evaluate_at_line(
            doppler_estimates,
            line_index=line_index,
            azimuth_times_s=combined_eta,
            slant_range_times_s=slant_range_times_s,
        )

    return (doppler_centroid_for_line,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 — Chuẩn bị block focus Stripmap

    Tính effective radar velocity (§9.10), azimuth FM rate (§9.11), overlap và
    chiều dài block focus (§9.12–§9.13), sau đó xác định support output hợp lệ
    cho Stripmap SLC (§8.3.1).
    """)
    return


@app.cell
def _(combined_eta, common, l0file, wavelength_m):
    # DAD §9.10: model effective radar velocity from the orbit state vectors.
    # One shared estimator feeds SRC, RCMC, and azimuth compression.
    velocity_estimator = common.effective_velocity.Estimator.from_level0_product(
        l0file, wavelength_m
    )
    # L0 packet velocities are quantised; smooth the L0 positions only for the
    # scene-wide worst-case matched-filter support calculation.
    layout_velocity_estimator = (
        common.effective_velocity.Estimator.from_level0_product(
            l0file, wavelength_m, smooth_positions=True
        )
    )
    # Scene này kết thúc 0,90 s sau epoch state-vector cuối của product.
    velocity_estimator.validate_time_coverage(
        combined_eta, max_extrapolation_s=1.0
    )
    return layout_velocity_estimator, velocity_estimator


@app.cell
def _(
    az_sample_freq,
    azimuth_processing,
    c,
    combined_eta,
    common_tau,
    doppler_centroid_for_line,
    geometry_dc_estimator,
    layout_velocity_estimator,
    np,
    range_sample_freq,
    raw_tau_segments,
    s6_parameters,
    velocity_estimator,
    wavelength_m,
):
    FOCUS_FFT_LEN = s6_parameters.FOCUS_FFT_LENGTH
    AZIMUTH_PROCESSING_BANDWIDTH_HZ = s6_parameters.FOCUS_AZIMUTH_BANDWIDTH_HZ
    slant_ranges_m = common_tau * c / 2.0
    # DAD §9.12-§9.13: tính A trên maximum far range của raw L0;
    # bandwidth, maxFdc, FFT length và extra overlap đến từ AUX_PP1.
    raw_range_extent_s = np.array([
        min(tau[0] for tau in raw_tau_segments),
        max(tau[-1] for tau in raw_tau_segments),
    ])
    raw_range_extent_m = raw_range_extent_s * c / 2.0
    focus_layout = azimuth_processing.processing_blocks.calculate_layout(
        len(combined_eta),
        raw_range_extent_m,
        combined_eta,
        layout_velocity_estimator,
        wavelength_m=wavelength_m,
        azimuth_sample_frequency_hz=az_sample_freq,
        processing_bandwidth_hz=AZIMUTH_PROCESSING_BANDWIDTH_HZ,
        max_doppler_centroid_hz=(
            s6_parameters.FOCUS_MAX_DOPPLER_CENTROID_HZ
        ),
        fft_length=FOCUS_FFT_LEN,
        extra_overlap_samples=s6_parameters.EXTRA_AZIMUTH_OVERLAP_SAMPLES,
    )
    # DAD §6.3.2 and §8.3.1: trim azimuth-filter and RCMC edge transients and
    # anchor the valid SLC on its zero-Doppler output timeline.
    output_geometry = azimuth_processing.processing_blocks.derive_output_geometry(
        slant_ranges_m,
        combined_eta,
        doppler_centroid_for_line,
        lambda line: geometry_dc_estimator.estimate(
            combined_eta[line], slant_ranges_m
        ),
        velocity_estimator,
        focus_layout,
        wavelength_m=wavelength_m,
        speed_of_light_mps=c,
        azimuth_sample_period_s=1.0 / az_sample_freq,
        range_sample_frequency_hz=range_sample_freq,
        processing_bandwidth_hz=AZIMUTH_PROCESSING_BANDWIDTH_HZ,
        fft_length=FOCUS_FFT_LEN,
        rcmc_kernel_length=s6_parameters.RCMC_KERNEL_LENGTH,
        rcmc_phases=s6_parameters.RCMC_PHASES,
        apply_coarse_bistatic_delay_correction=True,
        support_slant_ranges_m=raw_range_extent_m,
        support_doppler_centroid_for_line=lambda line: (
            doppler_centroid_for_line(line, raw_range_extent_s)
        ),
    )
    return (
        AZIMUTH_PROCESSING_BANDWIDTH_HZ,
        FOCUS_FFT_LEN,
        focus_layout,
        output_geometry,
        slant_ranges_m,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 — Azimuth pre-processing và azimuth processing

    Mỗi block đi đúng thứ tự Figure 6-1: azimuth zero-padding (§6.2.1), forward
    FFT (§6.2.3), SRC (§6.3.1), RCMC (§6.3.2), rồi azimuth compression
    (§6.3.4). Stripmap không có frequency/time UFR của TOPSAR.
    """)
    return


@app.cell
def _(
    AZIMUTH_PROCESSING_BANDWIDTH_HZ,
    CACHE_ROOT,
    FOCUS_FFT_LEN,
    PRI,
    SCENE_CACHE_KEY,
    azimuth_processing,
    c,
    cache,
    combined_eta,
    combined_range_cache,
    doppler_centroid_for_line,
    doppler_estimates,
    focus_layout,
    focus_source,
    input_identity,
    output_geometry,
    range_sample_freq,
    range_sample_period,
    s6_parameters,
    slant_ranges_m,
    velocity_estimator,
    wavelength_m,
):
    def _focus_to_file(destination):
        _source = cache.open_array(combined_range_cache)

        def _write_focus(output):
            # DAD Figure 6-1: §6.2.1 -> §6.2.3 -> §6.3.1 -> §6.3.2 -> §6.3.4.
            azimuth_processing.processing_blocks.focus_slc(
                _source,
                slant_ranges_m,
                combined_eta,
                doppler_centroid_for_line,
                velocity_estimator,
                focus_layout,
                wavelength_m=wavelength_m,
                speed_of_light_mps=c,
                azimuth_sample_period_s=PRI,
                range_sample_period_s=range_sample_period,
                range_sample_frequency_hz=range_sample_freq,
                processing_bandwidth_hz=AZIMUTH_PROCESSING_BANDWIDTH_HZ,
                fft_length=FOCUS_FFT_LEN,
                azimuth_time_correction_s=0.0,
                src_segment_samples=s6_parameters.SRC_SEGMENT_SAMPLES,
                rcmc_kernel_length=s6_parameters.RCMC_KERNEL_LENGTH,
                rcmc_phases=s6_parameters.RCMC_PHASES,
                output_geometry=output_geometry,
                output=output,
            )

        cache.write_memmap(
            destination,
            output_geometry.shape,
            _write_focus,
        )

    with cache.persistent(
        "focused-scene", f"{CACHE_ROOT}/{SCENE_CACHE_KEY}"
    ):
        input_identity, focus_source, doppler_estimates, combined_range_cache
        focused_cache_file = (
            f"{CACHE_ROOT}/{SCENE_CACHE_KEY}/focused-scene/data.npy"
        )
        _focus_to_file(focused_cache_file)
    focused_slc = cache.open_array(focused_cache_file)
    return (focused_slc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 — Internal SLC output (DAD §8.3.1)

    Chỉ giữ các sample còn đủ support của azimuth matched filter và RCMC.
    Level-1 post-processing, radiometric calibration và SAFE formatting nằm
    ngoài phạm vi workflow này.
    """)
    return


@app.cell
def _(alignment_summary, focus_layout, focused_slc, iq_biases):
    print("SLC shape:", focused_slc.shape)
    print("I/Q bias các segment:", iq_biases)
    print("Azimuth matched-filter support:", focus_layout.matched_filter_support_samples)
    print("Overlap focus:", focus_layout.overlap_samples, "lines")
    print("Azimuth block step:", focus_layout.step_samples, "lines")
    alignment_summary
    return


@app.cell
def _(colors, focused_slc, np, plt):
    _amplitude = np.abs(focused_slc[::20, ::20])
    _positive = _amplitude[_amplitude > 0]
    _vmin = np.percentile(_positive, 2)
    _vmax = np.percentile(_positive, 98)

    plt.figure(figsize=(12, 12), dpi=75)
    plt.title("Focused SLC — toàn scene")
    plt.imshow(
        _amplitude,
        origin="lower",
        cmap="viridis",
        norm=colors.LogNorm(vmin=_vmin, vmax=_vmax),
        aspect="auto",
    )
    plt.xlabel("Slant range (mỗi 20 samples)")
    plt.ylabel("Azimuth (mỗi 20 lines)")
    plt.show()
    return


@app.cell
def _(colors, focused_slc, np, plt):
    _amplitude = np.abs(focused_slc[9000:10500, 5500:6800])
    _positive = _amplitude[_amplitude > 0]
    _vmin = np.percentile(_positive, 2)
    _vmax = np.percentile(_positive, 98)

    plt.figure(figsize=(12, 12), dpi=75)
    plt.title("Focused SLC — chi tiết")
    plt.imshow(
        _amplitude,
        origin="lower",
        cmap="viridis",
        norm=colors.LogNorm(vmin=_vmin, vmax=_vmax),
        aspect="auto",
    )
    plt.xlabel("Slant range")
    plt.ylabel("Azimuth")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
