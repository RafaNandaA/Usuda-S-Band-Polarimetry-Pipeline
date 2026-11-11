# Apply a calibration table set to the stored per-window coherency and fold blocks, and write the calibrated products.
#
#   julia -t 4 calibrate/apply.jl CONFIG.toml [catalogue|fold|all]
#
# Reads   paths.work_dir/items/cat_N.h5 (one per catalogue window) and paths.work_dir/blocks/block_B.h5 (fold), written by
#         extract/extract.jl, and the table set in paths.tables_dir (written by calibrate/fit.py).
# Writes  paths.products_dir/CrabGRP_catalogue.h5      groups catalog, metadata, nebula_diag, raw_profiles
#         paths.products_dir/CrabGRP_FreqResolved.h5   the same + freq_resolved (per-channel Stokes and noise)
#         paths.products_dir/CrabGRP_fold.h5           mean main-pulse and LFIP profiles, sky and feed frame
# The tables the data were read with (paths.readtime_tables_dir) must be the ones given at read time, because the stored
# reference profiles and baseline cross moments were made with them.
using Base.Threads, Printf, Statistics
length(ARGS) >= 1 || error("usage: julia -t 4 calibrate/apply.jl CONFIG.toml [catalogue|fold|all]")
ENV["USUDAPOL_CONFIG"] = abspath(ARGS[1])
include(joinpath(@__DIR__, "lib", "apply_lib.jl"))
const WHAT = length(ARGS) >= 2 ? ARGS[2] : "all"
WHAT in ("catalogue", "fold", "all") || error("unknown product '$WHAT' (catalogue, fold or all)")
const PATHS = V2.CFG["paths"]
const RUN = PATHS["work_dir"]
const TDIR = PATHS["tables_dir"]
const ODIR = PATHS["products_dir"]; mkpath(ODIR)
const Trt = tables_from_module(V2, "readtime")
const T = load_tables(TDIR, Trt)
const TSHA = tables_sha256(TDIR)
const PROV = V2.provenance_string() * ". Calibration tables $(TDIR) (sha256 $(TSHA)), feed model alpha $(T.bwg_alpha), gamma_Az $(T.bwg_gamma)"
println(PROV)

const SKY_PA_NOTE = "PA sense: sky PA runs N through W (opposite to IAU). Catalogue PA_* are in the feed frame, sky PA = wrap90(PA - chi_BWG_deg). No absolute PA (the common R/L phase is not identifiable), so only PA differences are meaningful"
const V_NOTE = "Stokes V handedness is not tied to the sky (no calibrator scan). V zero point: system V = 0 over the nebula snapshots"
const FLUX_NOTE = "calibrated_arb: nebula-relative arbitrary units (no calibrator scan and no SEFD, so no Jy)"
const FILE_ATTRS = Dict{String,Any}(
    "calibration_stage" => "per-channel chain: bandpass, gain, differential gain gamma(t), R/L delay (phase ramp on L before channelisation, plus a residual per-channel phase), R/L phase, D-terms (kappa form), ellipticity (m_QV, m_UV), ionospheric RM. FreqResolved columns I/Q/U/V_chan* are after the D-terms and before the ellipticity and RM, feed frame",
    "calibration_tables_sha256" => TSHA, "calibration_table_dir" => TDIR,
    "dispersion_measure_pc_cm3" => V2.DM, "dedispersion_reference_MHz" => V2.f_ref_MHz,
    "delay_method" => "R/L delay applied as a sky-frequency phase ramp cis(2 pi f_sky tau_s) on L inside the dedispersion FFT, before channelisation, tau_s_ns = " * join(string.(round.(Trt.tau .* 1e9; digits=3)), "/"),
    "aperture_rule" => "3-sigma aperture: contiguous run of calibrated band-I samples above 3 sigma around the peak. Union aperture: 3-sigma runs whose own 1-us peak reaches 5 sigma within +-30 us of the main peak are merged, gaps included (union_flag = runs merged). Mean and sigma from the first 20 % of the 20-ms window. *_int are baseline-subtracted sums over the aperture",
    "time_resolution_us" => 1.0, "channel_width_MHz" => 1.0, "n_channels" => 128,
    "band_MHz" => "2194-2322 contiguous (4 subbands x 32 MHz, lower edges 2194/2226/2258/2290, channel index subband-major)",
    "frame" => "feed frame unless a name says sky. sky PA = wrap90(PA_feed - chi_BWG_deg), chi = q + alpha*El + gamma*Az with alpha $(T.bwg_alpha), gamma $(T.bwg_gamma)",
    "sign_conventions" => SKY_PA_NOTE * ". " * V_NOTE, "flux_scale" => FLUX_NOTE,
    "known_limitations" => "band sums leave out channel 32 of every subband and channel 1 of subband 2 (weights n_ch/n_kept, per-channel columns keep all channels). RM is not measured (degenerate with the R/L delay and phase). Ionospheric RM_ion(t) from a global ionosphere map is applied per window. Nanosecond structure is not resolved (1 us, 1 MHz). Per-pulse errors leave out the pulse x nebula cross term and the pulse self-noise. Raw profiles outside the stored span carry the read-time noise weighting",
    "creation_date" => string(Dates.now()))
"write the file-level attributes"
function stamp!(obj)
    A = attributes(obj)
    for (k, v) in FILE_ATTRS; haskey(A, k) && HDF5.delete_attribute(obj, k); A[k] = v; end
end
"per-dataset attributes: units, frame, calibration stage, description"
function describe!(parent, name; units, frame="feed", stage, desc)
    a = attributes(parent[name])
    for (k, v) in (("units", units), ("frame", frame), ("calibration_stage", stage), ("description", desc)); haskey(a, k) && HDF5.delete_attribute(parent[name], k); a[k] = v; end
end
const SNR10_DEF = "10-us running-boxcar S/N of the calibrated band I: z_i = (mean of 10 consecutive 1-us samples starting at bin i - mu_I) / (sigma_I / sqrt(10)), " *
    "band I over kept channels (weights n_ch/n_kept, /32), mu_I and sigma_I from the first 20 % of the 20-ms window. sigma/sqrt(10) assumes independent 1-us samples. " *
    "Column 1 = maximum over the window, column 2 = maximum over runs inside the peak window PK_LO..PK_HI (0.064-0.069 s)"

# ------------------------------------------------------------------------------------------------ catalogue
if WHAT in ("catalogue", "all")
    files = sort([f for f in readdir(joinpath(RUN, "items")) if startswith(f, "cat_") && endswith(f, ".h5")], by = f -> parse(Int, f[5:end-3]))
    N = length(files); R = Vector{Any}(undef, N)
    @threads :dynamic for i in 1:N
        R[i] = calibrate_window(joinpath(RUN, "items", files[i]), T, Trt)
    end
    profs = [r[1] for r in R]; rows = [r[2] for r in R]; freqs = [r[3] for r in R]; exs = [r[4] for r in R]; dgs = [r[5] for r in R]
    V2.write_archive(rows, profs, "catalogue", ODIR)
    FR.write_archive(rows, profs, freqs, "FreqResolved", ODIR)
    sn(f, k) = h5open(h -> read(h["snr/$k"]), joinpath(RUN, "items", f), "r")
    for p in (joinpath(ODIR, "CrabGRP_catalogue.h5"), joinpath(ODIR, "CrabGRP_FreqResolved.h5"))
        h5open(p, "r+") do h
            c = h["catalog"]
            for k in ("I_uncal", "R_uncal", "L_uncal", "I_cal_readtime")
                c["snr6_$k"] = reduce(hcat, [sn(f, k) for f in files])           # (6, N) -> h5py (N, 6)
                attributes(c["snr6_$k"])["columns"] = "1us_peak, run10, block10 (max over the window), then 1us_peak, run10, block10 (max over PK_LO..PK_HI)"
            end
            c["read_flag_bits"] = [h5open(x -> read(x["flag/bits"]), joinpath(RUN, "items", f), "r") for f in files]
            attributes(c["read_flag_bits"])["rule"] = "1 spiky interval, 2 1-ms power excess, 4 extreme 4-bit levels, 8 incomplete read (window-internal reference). Flagged, never dropped"
            c["rm_ion"] = [rm_ion_at(T, parse(Float64, string(r.sod))) for r in rows]
            c["ion_pa_shift_deg"] = [T.ion_sigma * rm_ion_at(T, parse(Float64, string(r.sod))) * rad2deg((V2.C_MPS / 2258e6)^2) for r in rows]
            c["snr10_I_cal_final"] = reduce(hcat, [[e.snr10[1], e.snr10[2]] for e in exs])          # (2, N) -> h5py (N, 2)
            c["snr10_I_cal_final_pos"] = reduce(hcat, [[e.snr10[3], e.snr10[4]] for e in exs])
            attributes(c["snr10_I_cal_final"])["definition"] = SNR10_DEF
            attributes(c["snr10_I_cal_final_pos"])["definition"] = "1-based window bin of the first of the 10 samples of the run that gives the S/N in the same column of snr10_I_cal_final"
            # scalar columns derived from the vector quantities above
            c["sigma_Q"] = [e.sigq for e in exs]; c["sigma_U"] = [e.sigu for e in exs]; c["nb"] = [e.nb for e in exs]
            c["sigma_R_band"] = [e.sigR for e in exs]; c["sigma_L_band"] = [e.sigL for e in exs]
            c["ipk_run"] = [h5open(x -> Int(read(x["scalars/ipk_run"])), joinpath(RUN, "items", f), "r") for f in files]
            c["I_SN_cal_run10_pk"] = [e.snr10[2] for e in exs]
            c["I_SN_uncal_run10_pk"] = [sn(f, "I_uncal")[5] for f in files]; c["SN_R_run10_pk"] = [sn(f, "R_uncal")[5] for f in files]; c["SN_L_run10_pk"] = [sn(f, "L_uncal")[5] for f in files]
            c["mon_power_excess_max"] = [h5open(x -> read(x["flag/maxdev_F2"]), joinpath(RUN, "items", f), "r") for f in files]
            c["mon_levels_ratio_max"] = [h5open(x -> read(x["flag/maxratio_F3"]), joinpath(RUN, "items", f), "r") for f in files]
            h["nebula_diag"]["VI_neb_preGamma"] = [e.VI_neb_preGamma for e in exs]
            c["offline_aperture_changed"] = [Int(d["ap3_changed"]) + 2Int(d["union_changed"]) + 4Int(d["peak_changed"]) for d in dgs]
            c["offline_span_ok"] = [Int(d["span_ok"]) + 2Int(d["dyn_ok"]) for d in dgs]
            A = attributes(h["metadata"])
            for k in ("provenance", "parallactic_corrected", "polarization_calibrated"); haskey(A, k) && HDF5.delete_attribute(h["metadata"], k); end
            A["provenance"] = PROV; A["calibration_tables_sha256"] = TSHA; A["calibration_tables"] = TDIR
            A["polarization_calibrated"] = "bandpass + gain + gamma(t) + R/L delay and phase + D-terms (kappa) + ellipticity (Q,U -> V) + ionospheric RM"
            A["parallactic_corrected"] = "catalog PA_* are feed frame. sky PA = wrap90(PA - nebula_diag/chi_BWG_deg)"
            A["DM_uncertainty"] = "+- 0.0002 (stat) +- 0.0004 (sys) pc cm^-3, scattering 0 to -0.0005"
            stamp!(h)
            describe!(c, "snr6_I_uncal", units="power (uncalibrated counts^2, 4 subbands x 32 ch summed)", stage="uncalibrated", desc="S/N of band I: 1us_peak, run10, block10 (max over the window), the same three over PK_LO..PK_HI")
            describe!(c, "snr6_R_uncal", units="power (uncalibrated)", stage="uncalibrated", desc="S/N of the R hand, as snr6_I_uncal")
            describe!(c, "snr6_L_uncal", units="power (uncalibrated)", stage="uncalibrated", desc="S/N of the L hand, as snr6_I_uncal")
            describe!(c, "snr6_I_cal_readtime", units="calibrated_arb", stage="read-time tables", desc="S/N of the calibrated band I with the read-time tables")
            describe!(c, "read_flag_bits", units="bit mask (1 spiky interval, 2 power, 4 levels, 8 incomplete)", frame="n/a", stage="n/a", desc="quality flags from the reader, flagged and never dropped")
            describe!(c, "offline_aperture_changed", units="bit mask (1 3-sigma aperture, 2 union, 4 peak moved vs the read-time apertures)", frame="n/a", stage="n/a", desc="apertures measured with these tables differ from the read-time ones")
            describe!(c, "offline_span_ok", units="bit mask (1 apertures inside the stored span, 2 union +-50 us inside)", frame="n/a", stage="n/a", desc="coverage of the stored dynamic-spectrum span")
            A["units_note"] = "catalog *_int: " * FLUX_NOTE * ". LI/VI/PI in percent, PA in deg, widths in us. Each catalog dataset carries its own units attribute."
            A["sign_conventions"] = SKY_PA_NOTE * ". " * V_NOTE
            A["calibration_note"] = "raw_profiles inside the stored span (union +- 70 us) are exact for this table set. Outside it they are the read-time profile plus the exact mean shift. " *
                "sigma_* use the exact cross moments and the per-channel 4x4 blocks of the baseline covariance."
        end
    end
    h5open(joinpath(ODIR, "CrabGRP_FreqResolved.h5"), "r+") do h
        fr = h["freq_resolved"]; Np = N
        put3(nm, f) = (M = zeros(128, 4, Np); for i = 1:Np; M[:, :, i] .= permutedims(f(exs[i]), (2, 1)); end; fr[nm] = M)
        put3("Stokes_chan_cal", e -> e.fcal_on); put3("Stokes_chan_cal_union", e -> e.fcal_union)
        put3("Stokes_off_chan_cal", e -> e.fcal_off); put3("Stokes_base_chan_cal", e -> e.fcal_base)
        put3("sigma_chan_cal", e -> e.sig_full); put3("sigma_chan_preellip", e -> e.sig_pre)
        rc(nm, f) = (M = zeros(128, 4, Np); for i = 1:Np; M[:, :, i] .= permutedims(reshape(f(exs[i]), 4, 128), (2, 1)); end; fr[nm] = M)
        rc("RC_on_uncal", e -> e.RC_on); rc("RC_on_union_uncal", e -> e.RC_on_union); rc("RC_off_uncal", e -> e.RC_off); rc("RC_base_uncal", e -> e.RC_base)
        # per-channel baseline mean of I and per-sample variances (before the ellipticity), h5py shape (Np, 128)
        fr["muI_chan_calib"] = permutedims(reduce(hcat, [e.fcal_base[1, :] for e in exs]), (2, 1))
        for (k, nm) in enumerate(("I", "Q", "U", "V")); fr["sig2$(nm)_chan_calib"] = permutedims(reduce(hcat, [e.sig_pre[k, :] .^ 2 for e in exs]), (2, 1)); end
        fr["chi_deg"] = [e.chi for e in exs]; fr["W_ap3"] = [e.W[1] for e in exs]; fr["W_union"] = [e.W[2] for e in exs]
        attributes(fr)["note_calibrated_columns"] = "Stokes_*_cal: fully calibrated per channel, feed frame, h5py shape (Np, 4 = I,Q,U,V, 128 subband-major), " *
            "not /32 (sum over channels / 32 = band value). Stokes_chan_cal = 3-sigma aperture minus W x first-20% mean. sigma_* = per-sample per-channel " *
            "baseline sigma. RC_*_uncal = uncalibrated coherency (RR, LL, Re RL, Im RL) of the aperture sums, off-pulse mean and baseline mean, so the calibration can be redone without the raw data."
        for nm in ("Stokes_chan_cal", "Stokes_chan_cal_union", "Stokes_off_chan_cal", "Stokes_base_chan_cal")
            describe!(fr, nm, units="calibrated_arb per channel (not /32)", stage="full calibration", desc="per-channel Stokes (I,Q,U,V) of the " * (occursin("union", nm) ? "union aperture" : occursin("off", nm) ? "off-pulse mean" : occursin("base", nm) ? "first-20% baseline mean" : "3-sigma aperture") * ", baseline-subtracted for the aperture sums")
        end
        for nm in ("sigma_chan_cal", "sigma_chan_preellip"); describe!(fr, nm, units="calibrated_arb per channel per 1-us sample", stage=nm == "sigma_chan_cal" ? "full calibration" : "before the ellipticity", desc="per-sample baseline sigma of each channel Stokes (first 20 %)"); end
        for nm in ("RC_on_uncal", "RC_on_union_uncal", "RC_off_uncal", "RC_base_uncal"); describe!(fr, nm, units="uncalibrated counts^2 (RR, LL, Re RL, Im RL)", frame="instrument", stage="uncalibrated", desc="aperture sum or mean of the uncalibrated per-channel coherency"); end
        describe!(fr, "chi_deg", units="deg", frame="n/a", stage="n/a", desc="feed rotation chi = q + alpha*El + gamma*Az, sky PA = wrap90(PA_feed - chi)")
        stamp!(fr)
    end
    @printf("catalogue: %d windows, apertures moved with these tables: 3-sigma %d, union %d, peak %d, inside the stored span %d\n", N,
            count(d -> d["ap3_changed"], dgs), count(d -> d["union_changed"], dgs), count(d -> d["peak_changed"], dgs), count(d -> d["span_ok"], dgs))
end

# ------------------------------------------------------------------------------------------------ fold
# Each block file holds the uncalibrated per-channel coherency summed over the rotations of 60 s, in 125 bins of 16 us for
# the main-pulse, off-pulse and LFIP windows. gamma(t), RM_ion and the feed-angle phasor are applied as block means, which
# is exact to well below the noise because they change by << 1 % within 60 s.
"calibrated band profiles (4, 125) of one block accumulator with per-rotation inverse gammas igs (powers 0, 2, 1, 1 on RR, LL, RL)"
function fold_cal(M1, RC16, igs)
    m1 = mean(igs); m2 = mean(abs2, igs)
    X = Float64.(RC16); out = zeros(4, size(X, 4))
    @inbounds for j = 1:size(X, 4), s = 1:NSB, c = 1:NCH, k = 1:4
        out[k, j] += (M1[k, 1, c, s] * X[1, c, s, j] + M1[k, 2, c, s] * m2 * X[2, c, s, j] +
                      M1[k, 3, c, s] * m1 * X[3, c, s, j] + M1[k, 4, c, s] * m1 * X[4, c, s, j])
    end
    out ./ NCH
end
function fold_block(path, T)
    h = h5open(path, "r")
    fl = read(h["rot/flag"]); li = read(h["rot/listed"]); gp = read(h["rot/in_list_gap"]); utc = read(h["rot/utc"])
    q = read(h["rot/q_deg"]); el = read(h["rot/el_deg"]); az = read(h["rot/az_deg"])
    rmb = isempty(utc) ? 0.0 : mean(rm_ion_at(T, u) for u in utc)
    M1 = wscale(chanmats(T, 1.0; rmw = T.RM + T.ion_sigma * rmb)[1], T.chw)
    res = Dict{String,Any}("block" => read(attributes(h)["block"]))
    # 'all' = every unflagged rotation, 'free' = also without a listed giant pulse and outside the gaps of the lists
    for (set, sel) in (("all", fl .== 0), ("free", (fl .== 0) .& (li .== 0) .& (gp .== 0)))
        n = count(sel); res["n_$set"] = n
        n == 0 && continue
        igs = [1.0 / diffgain(T, u) for u in utc[sel]]
        χ = q[sel] .+ T.bwg_alpha .* el[sel] .+ T.bwg_gamma .* az[sel]
        Z = mean(cis.(-2 .* deg2rad.(χ))); res["Z_$set"] = Z; res["e2chi_sum_$set"] = sum(cis.(2 .* deg2rad.(χ)))
        for w in ("MP", "OFF", "LFIP")
            res["$set/$w"] = fold_cal(M1, read(h["$set/$w/RC16"]), igs)
        end
    end
    close(h); res
end
if WHAT in ("fold", "all") && isdir(joinpath(RUN, "blocks"))
    bf = sort([f for f in readdir(joinpath(RUN, "blocks")) if endswith(f, ".h5")], by = f -> parse(Int, f[7:end-3]))
    if !isempty(bf)
        B = Vector{Any}(undef, length(bf))
        @threads :dynamic for i in eachindex(bf); B[i] = fold_block(joinpath(RUN, "blocks", bf[i]), T); end
        h5open(joinpath(ODIR, "CrabGRP_fold.h5"), "w") do h
            for set in ("all", "free")
                tot = Dict(w => zeros(6, 125) for w in ("MP", "LFIP")); nrot = 0; e2 = 0.0im
                for w in ("MP", "OFF", "LFIP")
                    h["$set/per_block/$w"] = cat([get(b, "$set/$w", fill(NaN, 4, 125)) for b in B]..., dims=3)   # (4,125,nb) -> h5py (nb,125,4)
                end
                for b in B
                    haskey(b, "$set/MP") || continue
                    nrot += b["n_$set"]; e2 += b["e2chi_sum_$set"]
                    off = vec(mean(b["$set/OFF"], dims=2))          # nebula + system, subtracted per block before the rotation to the sky
                    for w in ("MP", "LFIP")
                        P = b["$set/$w"] .- off; L = (P[2, :] .+ im .* P[3, :]) .* b["Z_$set"]
                        tot[w][1, :] .+= P[1, :]; tot[w][2, :] .+= real.(L); tot[w][3, :] .+= imag.(L); tot[w][4, :] .+= P[4, :]
                        tot[w][5, :] .+= P[2, :]; tot[w][6, :] .+= P[3, :]
                    end
                end
                for w in ("MP", "LFIP")
                    h["$set/$w/profile"] = tot[w]
                    attributes(h["$set/$w/profile"])["rows"] = "I, Q_sky, U_sky, V, Q_feed, U_feed (h5py columns), 125 x 16-us bins, off-pulse window mean subtracted per block, sky = (Q+iU) x block mean of e^{-2i chi}"
                end
                h["$set/n_rot"] = nrot; h["$set/smear_abs_mean_e2ichi"] = abs(e2) / max(nrot, 1)
            end
            h["blocks"] = [b["block"] for b in B]; h["n_all_per_block"] = [b["n_all"] for b in B]; h["n_free_per_block"] = [b["n_free"] for b in B]
            attributes(h)["provenance"] = PROV; stamp!(h)
            for set in ("all", "free"), w in ("MP", "LFIP"); describe!(h, "$set/$w/profile", units="calibrated_arb summed over rotations, 125 x 16-us bins", frame="sky (Q,U rows 2,3) and feed (rows 5,6)", stage="full calibration, off-pulse window mean subtracted per block", desc="folded profile rows I, Q_sky, U_sky, V, Q_feed, U_feed"); end
        end
        println("fold: $(length(bf)) blocks -> ", joinpath(ODIR, "CrabGRP_fold.h5"))
    end
end
