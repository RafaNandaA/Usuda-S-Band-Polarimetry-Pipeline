# usuda-polarimetry

This pipeline turns dual-circular 4-bit voltage recordings from the Usuda 64-m antenna (S band, 2194-2322 MHz) into a calibrated full-Stokes (I, Q, U, V) catalogue of Crab giant pulses, a calibrated mean-pulse fold, and the polarization statistics built on them. The off-pulse Crab Nebula fills the beam and is the polarization calibrator, so no calibrator scan is needed. The stages are written for one Crab track. The parts specific to the Crab are the timing (taken from the detection lists) and the source position used for the parallactic angle (`extract/lib/kernel.jl`).

The pipeline was used for 3632 listed giant pulses from one 5-h track on 2026-02-12. No data are included here.

## Data format

- One raw file per track, no header. Each little-endian `UInt32` word packs 8 simultaneous 4-bit samples (offset binary, value - 7.5), nibbles CH0-3 = right-circular (R) subbands 0-3 and CH4-7 = left-circular (L) subbands 0-3.
- 64 Msps real sampling per subband, four 32-MHz subbands with lower edges 2194, 2226, 2258 and 2290 MHz, so the band is contiguous from 2194 to 2322 MHz.
- Two per-hand detection lists (RHCP and LHCP, 10-us integration, list S/N >= 5) with columns GRP_ID, detection time (UTC and TDB seconds of day), rotation number Nseq, rotation phase and S/N. The lists also give the timing model, so no ephemeris is needed.

## Requirements

- Julia 1.12 with FFTW and HDF5 (plus standard libraries). From this folder: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`. Start Julia with threads (`-t 4`, or `-t 5,1` for the reader).
- Python 3.11 or later with numpy, scipy, h5py and matplotlib (`pip install -r requirements.txt`). The calibration fit also needs `ppigrf` for the geomagnetic field in the ionosphere correction.
- The raw file is only read. Reading the catalogue windows of one track is about 19 GB (3632 windows of 20 ms), and the nebula snapshots and fold rotations add to that.

## Configuration

Every stage takes one TOML file. `config/example.toml` lists every setting with a short reason: paths (raw file, detection lists, work, tables, products, outputs), DM and reference frequency, subband edges, channeliser size, aperture parameters, the selection cut, the feed model, the measured R/L delay, the R-chain gain step and the nebula and fold plans. Relative paths are read relative to the config file. Copy it, edit the paths, and pass the copy to every command below (`CFG` = path of your copy).

## Running the stages

Run the commands from this folder, in this order. The extract stage is run twice because the calibration tables are fitted on the nebula snapshots, and the pulse windows are then read with those tables.

1. Plan the reads (no raw data needed):
   `julia --project=. extract/extract.jl CFG plan`
   This fits the timing model to the detection lists and writes `work_dir/items.tsv` (one 20-ms window per listed pulse with max(S/N_R, S/N_L) >= 6, and the fold rotations), `nebula_items.tsv`, `fold_blocks.tsv` and `timing_model.json`.
2. Read the nebula snapshots:
   `julia --project=. -t 4 extract/extract.jl CFG nebula`
   This writes the off-pulse coherency snapshots (3000 x 156 ms over the track, listed pulses masked along their dispersed track) to `paths.nebula_file`, and the 1-s series around the R-chain gain step to `paths.gain_step_file`.
3. Fit the calibration:
   `python3 calibrate/fit.py CFG`
   This writes the table set to `tables_dir` (bandpass, differential gain gamma(t), D-terms, ellipticity, ionospheric RM track, `scalars.txt`) and `inputs.json` for the analysis. It takes less than a minute.
4. Read the pulse windows and fold rotations:
   `julia --project=. -t 5,1 extract/extract.jl CFG windows`
   Each catalogue window is stored uncalibrated in `work_dir/items/cat_N.h5`, and each 60-s fold block in `work_dir/blocks/block_B.h5`. A stopped run continues from where it stopped when the same command is run again.
5. Apply the calibration:
   `julia --project=. -t 4 calibrate/apply.jl CFG all`
   This writes the catalogue, the per-channel catalogue and the fold to `products_dir` in about a minute. Because the store is uncalibrated, a new table set only needs this step again (`catalogue` or `fold` instead of `all` makes one product).
6. Polarization statistics:
   `python3 analyze/run.py CFG`
   The self-noise sums are computed first with `analyze/selfnoise.jl` if they are missing. The output is `analysis_dir/NUMBERS.json` and `run_log.txt`, where every decision rule is printed with its outcome. With 20000 bootstrap draws the run takes about 5 minutes.
7. Mean-pulse fold:
   `python3 fold/fold.py CFG`
   This writes `analysis_dir/fold/fold_numbers.json` and `fold_log.txt` (profile L/I, V/I, PA and its slope, feed-frame controls, giant-pulse share of the mean flux).
8. Figures:
   `python3 figures/make_figures.py CFG`
   One panel per file, written to `figures_dir` as PDF and PNG.

The estimator unit tests need no data: `python3 tests/test_estimators.py`.

## Processing chain

1. Coherent dedispersion of each hand in a long FFT (reference 2322 MHz), so the dispersion sweep of about 1.3 ms per subband is removed.
2. The R/L delay (about -90 ns, near 6 samples) is applied as a phase ramp on L inside the dedispersion FFT, before channelisation. The same delay applied after the 64-point channeliser would decorrelate the cross-hand and scale Q, U and L by about 0.92.
3. A 64-point FFT channeliser gives 32 channels of 1 MHz per subband at 1-us sampling, with no time averaging.
4. Calibration per channel, in this order: amplitude bandpass, differential gain gamma(t) (settling, R-chain gain step and a band-common fast term), residual R/L delay as a per-channel phase, R/L phase per subband, D-terms, ellipticity (Q,U -> V, fitted per subband and applied per channel) and the ionospheric Faraday rotation.
5. Integration over the on-pulse aperture. The union aperture (primary) is the 3-sigma run around the peak plus any 3-sigma run within 30 us whose own 1-us peak reaches 5 sigma. The baseline is the first 20 % of the 20-ms read.
6. Feed rotation is applied in the analysis: sky PA = wrap90(PA - chi), chi = q - El, with q the parallactic angle. The feed room turns in azimuth and does not tip, so there is no azimuth term.

## Outputs

Intensities are in nebula-relative arbitrary units (`calibrated_arb`), angles in degrees, times in seconds of day (UTC) or microseconds. Most datasets carry a `units` attribute, and every file carries its provenance, sign conventions and known limitations as attributes.

`CrabGRP_catalogue.h5`, one row per listed window (3632 for the 2026-02-12 track):
- `catalog`: integrated Stokes over the union aperture `I_int_union, Q_int_union, U_int_union, V_int_union` (feed frame, baseline subtracted), the 3-sigma-run versions `I_int ... V_int`, peak values `*_peak`, fractions in percent (`LI_*`, `VI_*`, `PI_*`) and feed-frame `PA_*`, widths `width_union_us`, `width_us`, `W_eq_us`, aperture bins `union_lo/hi`, `ap3_lo/hi`, `peak_bin`, noise `baseline_mu_I`, `sigma_I`, `sigma_Q`, `sigma_U`, `sigma_V`, `nb`, timing `sod_utc_s`, `nseq`, `phase`, `MP_IP_flag` (0 main pulse, 1 interpulse, -1 other), `valid`, detection S/N `detsnr`, `I_SN_cal_run10_pk`, `SN_R_run10_pk`, `SN_L_run10_pk`, ionosphere `rm_ion`, `ion_pa_shift_deg`, and quality bits `read_flag_bits` (1 interference interval, 2 power, 4 extreme 4-bit levels, 8 incomplete read). Flagged windows are kept.
- `nebula_diag`: per-window geometry `el_deg`, `az_deg`, `q_deg`, `chi_BWG_deg`, and the off-pulse nebula `I_neb`, `L_neb`, `V_neb`, `PA_neb_deg`, their values before the D-term correction, and `VI_neb_preGamma`.
- `raw_profiles`: calibrated band I, Q, U, V over +-750 us around the peak at 1 us (not baseline subtracted), and `peak_index`.

`CrabGRP_FreqResolved.h5`: the same groups plus `freq_resolved`, with per-channel Stokes over each aperture (`I_chan_union` ... `V_chan_union`, 128 channels, before the ellipticity and ionosphere corrections), fully calibrated per-channel Stokes (`Stokes_chan_cal*`), per-channel baseline means and variances, and `freq_MHz`.

`CrabGRP_fold.h5`: mean main-pulse and LFIP profiles in 16-us bins (I, Q and U in the sky and feed frames, V) for all rotations (`all`) and for rotations without a listed pulse (`free`), with the per-block data used for the bootstrap errors.

`NUMBERS.json` (analysis): the selected sample, the census of L/I and V/I, the ratio of mean-square linear to circular polarization with its ellipticity-conditional limits, the total degree, the preferred PA direction and its rate test against the feed angle, the LFIP pulses alone, the aligned fraction, the dependence on duration at fixed brightness, the element bound, the sign balance of V over the zero-point range, the energy distribution, and every robustness variant. Each entry holds its value and interval, and each decision is listed with its rule and outcome.

## Calibration assumptions and limits

- The off-pulse Crab Nebula is the only calibrator. There is no calibrator scan and no SEFD, so intensities are not in Jy.
- System V = 0 is defined on the nebula (gamma(t) has mean 1 over the snapshots used for the bandpass). The handedness of V is not tied to the sky, and the pulse V zero point is known only within a range that the analysis carries through the sign test.
- There is no absolute PA. The common R/L phase cannot be separated from the feed model, so only PA differences are meaningful. The PA runs from N through W, opposite to the IAU sense.
- The rotation measure is not measured. Over 128 MHz it is degenerate with the per-subband R/L delay and phase, so RM is held at 0 and only the ionospheric part (from the GIM and IGRF) is removed.
- Channel 32 of every subband and channel 1 of subband 2 fail the interior response test and are left out of band sums. Per-channel columns keep them.
- The 4-bit samples are used without a quantisation correction. Bright pulses raise the raw power by 2 % (median) and 20 % (largest), so the analysis keeps a Q-guard variant that leaves out windows above a 10 % excess.
- The channeliser framing of each hand adds a per-pulse V_int term of about 0.4 % of I (rms, mean zero), which the sign test includes in the noise.
- The catalogue errors (`*_err`) leave out on-pulse self-noise and the pulse x nebula cross term. The analysis adds both.
- Nanosecond structure in the pulses is never resolved at 1-us sampling.

## Citation

If you use this pipeline, please cite: [citation to be added].

## License

[License to be added.]
