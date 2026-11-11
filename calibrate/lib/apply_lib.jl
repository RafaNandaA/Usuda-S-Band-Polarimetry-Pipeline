# Calibration of the stored per-window coherency with a table set.
#
# The reader stores, for every catalogue window, the uncalibrated per-channel coherency x = (RR, LL, Re RL, Im RL) of the
# apertures, the baseline and the off-pulse region, the baseline second moments, and the per-sample coherency over the union
# aperture +- 70 us. Every calibration term (bandpass, gain, gamma(t), R/L phase, residual delay, D-terms, ellipticity, RM)
# is linear in x and constant within one window, so calibrate(sum x) = sum calibrate(x). Per channel the map is a 4x4 real
# matrix, built here by running the kernel chain on the four unit vectors. The R/L delay ramp and the DM act before
# channelisation, so they are fixed at read time, and a table set with another delay ramp is refused.
#
# Include this file after ENV["USUDAPOL_CONFIG"] is set. It loads the kernels as modules V2 (band catalogue) and FR
# (per-channel catalogue) from extract/lib.
using Printf, Statistics, HDF5, Dates, LinearAlgebra, SHA
const KERNEL_DIR = normpath(joinpath(@__DIR__, "..", "..", "extract", "lib"))
isdefined(Main, :V2) || @eval Main module V2; include(joinpath($KERNEL_DIR, "kernel.jl")); end
isdefined(Main, :FR) || @eval Main module FR; include(joinpath($KERNEL_DIR, "kernel_freq.jl")); end
const NCH = V2.NCHAN; const NSB = V2.n_subbands
const OBSDATE = Date(string(V2.cfgget(V2.CFG, "observation", "date", "2026-02-12")))

# ------------------------------------------------------------------------------------------------ table sets
mutable struct Tables
    name::String
    bpR::Union{Nothing,Vector{Vector{Float64}}}; bpL::Union{Nothing,Vector{Vector{Float64}}}
    gR::Vector{Float64}; gL::Vector{Float64}
    dg_sod::Vector{Float64}; dg_gamma::Vector{Float64}
    phi::Vector{Float64}; tau::Vector{Float64}
    dR::Vector{Vector{ComplexF64}}; dL::Vector{Vector{ComplexF64}}
    mQV::Vector{Vector{Float64}}; mUV::Vector{Vector{Float64}}
    RM::Float64; bwg_alpha::Float64; bwg_gamma::Float64
    # residual R/L delay as a per-channel phase on L, band-sum channel weights, ionospheric RM_ion(sod) and its sign in the
    # pipeline PA sense
    tres::Vector{Float64}; chw::Matrix{Float64}; ion_sod::Vector{Float64}; ion_rm::Vector{Float64}; ion_sigma::Float64
end
"the tables a loaded kernel module uses (the read-time tables)"
tables_from_module(M::Module, name="readtime") = Tables(name, deepcopy(M.bandpass_R), deepcopy(M.bandpass_L),
    copy(Float64.(M.gain_R)), copy(Float64.(M.gain_L)), copy(M.diffgain_sod), copy(M.diffgain_gamma),
    copy(M.phase_offset_RL), copy(M.delay_RL), deepcopy(M.d_R), deepcopy(M.d_L), deepcopy(M.mQV), deepcopy(M.mUV),
    Float64(M.RM), Float64(M.BWG_ALPHA), Float64(M.BWG_GAMMA), copy(Float64.(M.tau_resid_RL)), copy(Float64.(M.chan_w)), Float64[], Float64[], 0.0)
"RM_ion(sod) of a table set (rad m^-2, IAU sign), linear interpolation, 0 when the set has no ionospheric track"
function rm_ion_at(T::Tables, sod::Real)
    isempty(T.ion_sod) && return 0.0
    sod <= T.ion_sod[1] && return T.ion_rm[1]
    sod >= T.ion_sod[end] && return T.ion_rm[end]
    i = searchsortedlast(T.ion_sod, sod); w = (sod - T.ion_sod[i]) / (T.ion_sod[i+1] - T.ion_sod[i])
    return T.ion_rm[i] * (1 - w) + T.ion_rm[i+1] * w
end
"RM applied for a window at sod: fitted RM (0) + sigma_ion * RM_ion(sod)"
rm_total(T::Tables, sod::Real) = T.RM + T.ion_sigma * rm_ion_at(T, sod)
"weight the band-sum matrices (k, i, c, s) by the channel weights (c, s)"
function wscale(M, W::Matrix{Float64})
    out = similar(M)
    for s in 1:size(M, 4), c in 1:size(M, 3); out[:, :, c, s] .= M[:, :, c, s] .* W[c, s]; end
    out
end
"channel weights from a list of dropped (s, c): n_ch/n_kept for kept channels of a subband, 0 for dropped, so a band sum
still divides by 32"
function weights_from_drop(drop::Vector{Tuple{Int,Int}})
    W = ones(NCH, NSB)
    for s in 1:NSB
        dr = [c for (ss, c) in drop if ss == s]; isempty(dr) && continue
        for c in 1:NCH; W[c, s] = c in dr ? 0.0 : NCH / (NCH - length(dr)); end
    end
    W
end
"gamma(t) of a table set: linear interpolation, clamped at the ends (the kernel's diffgain_at)"
function diffgain(T::Tables, sod::Real)
    isempty(T.dg_sod) && return 1.0
    sod <= T.dg_sod[1] && return T.dg_gamma[1]
    sod >= T.dg_sod[end] && return T.dg_gamma[end]
    i = searchsortedlast(T.dg_sod, sod)
    t0 = T.dg_sod[i]; t1 = T.dg_sod[i+1]; g0 = T.dg_gamma[i]; g1 = T.dg_gamma[i+1]
    return g0 + (g1 - g0) * (sod - t0) / (t1 - t0)
end
"load a table set (the four CSV tables, scalars.txt and rm_ion_track.csv written by calibrate/fit.py); missing entries keep
the values of `base`"
function load_tables(dir::String, base::Tables; name=basename(normpath(dir)))
    T = deepcopy(base); T.name = name
    rd(f) = [split(l, ",") for l in eachline(joinpath(dir, f)) if !(isempty(l) || startswith(l, "#") || startswith(l, "subband") || startswith(l, "sod"))]
    if isfile(joinpath(dir, "gainbandpass_cal_M64.csv"))
        T.bpR = [Float64[] for _ = 1:NSB]; T.bpL = [Float64[] for _ = 1:NSB]
        for p in rd("gainbandpass_cal_M64.csv"); s = parse(Int, p[1]); push!(T.bpR[s], parse(Float64, p[4])); push!(T.bpL[s], parse(Float64, p[5])); end
    end
    if isfile(joinpath(dir, "diffgain_time_M64.csv"))
        T.dg_sod = Float64[]; T.dg_gamma = Float64[]
        for p in rd("diffgain_time_M64.csv"); push!(T.dg_sod, parse(Float64, p[1])); push!(T.dg_gamma, parse(Float64, p[2])); end
    end
    if isfile(joinpath(dir, "dterm_cal_python_M64.csv"))
        for p in rd("dterm_cal_python_M64.csv")
            s = parse(Int, p[1]); c = parse(Int, p[2])
            T.dR[s][c] = complex(parse(Float64, p[4]), parse(Float64, p[5])); T.dL[s][c] = complex(parse(Float64, p[6]), parse(Float64, p[7]))
        end
    end
    if isfile(joinpath(dir, "mueller_vleak_M64.csv"))
        for p in rd("mueller_vleak_M64.csv"); s = parse(Int, p[1]); c = parse(Int, p[2]); T.mQV[s][c] = parse(Float64, p[3]); T.mUV[s][c] = parse(Float64, p[4]); end
    end
    sf = joinpath(dir, "scalars.txt")
    if isfile(sf)
        for l in eachline(sf)
            (isempty(strip(l)) || startswith(l, "#") || startswith(l, "drop_channels")) && continue
            k, v = strip.(split(l, "=", limit=2)); x = parse.(Float64, split(v, ","))
            k == "phase_offset_RL" && (T.phi = x); k == "delay_RL" && (T.tau = x); k == "RM" && (T.RM = x[1])
            k == "gain_R" && (T.gR = x); k == "gain_L" && (T.gL = x); k == "bwg_alpha" && (T.bwg_alpha = x[1]); k == "bwg_gamma" && (T.bwg_gamma = x[1])
            k == "tau_resid_RL" && (T.tres = x); k == "ion_sigma" && (T.ion_sigma = x[1])
        end
        for l in eachline(sf)
            startswith(l, "drop_channels") || continue
            v = strip(split(l, "=", limit=2)[2])
            T.chw = v == "none" ? ones(NCH, NSB) : weights_from_drop([(parse(Int, split(q, ":")[1]), parse(Int, split(q, ":")[2])) for q in split(v, ",")])
        end
    end
    if isfile(joinpath(dir, "rm_ion_track.csv"))
        T.ion_sod = Float64[]; T.ion_rm = Float64[]
        for l in eachline(joinpath(dir, "rm_ion_track.csv"))
            (isempty(l) || startswith(l, "#") || startswith(l, "sod")) && continue
            p = split(l, ","); push!(T.ion_sod, parse(Float64, p[1])); push!(T.ion_rm, parse(Float64, p[2]))
        end
    end
    T.tau == base.tau || error("table set $dir has delay_RL $(T.tau), the data were read with $(base.tau): the delay ramp acts before channelisation, so the raw data must be read again")
    T
end
"sha256 of the table files in a directory (identifies the table set in the product attributes)"
function tables_sha256(dir::String)
    fs_ = ("gainbandpass_cal_M64.csv", "diffgain_time_M64.csv", "dterm_cal_python_M64.csv", "mueller_vleak_M64.csv", "scalars.txt", "rm_ion_track.csv")
    bytes2hex(sha256(join([read(joinpath(dir, f), String) for f in fs_ if isfile(joinpath(dir, f))])))
end

# ------------------------------------------------------------------------------------------------ per-channel matrices
"the calibration chain for one channel on the coherency (RR, LL, X + iY). Returns (I, Q, U, V) fully calibrated,
(I, Q, U, V) after the D-terms and before the ellipticity and RM (per-channel catalogue stage), and (Q, U, V) before the D-terms"
function calib_point(T::Tables, s::Int, c::Int, ig::Float64, RR, LL, X, Y; rmw::Float64=T.RM)
    a = T.bpR === nothing ? 1.0 : 1.0 / max(T.bpR[s][c], 1e-12)
    b = T.bpL === nothing ? 1.0 : 1.0 / max(T.bpL[s][c], 1e-12)
    a /= T.gR[s]; b /= T.gL[s]; b *= ig
    fchm = V2.fmin_sub[s] + c * (V2.fs / V2.M_chan / 1e6)
    ph = exp(+im * (T.phi[s] + 2π * fchm * 1e6 * T.tres[s]))                 # phi_s + 2 pi f_ch tau_resid on L
    rr = a^2 * RR; ll = b^2 * LL; ρ = a * b * conj(ph) * complex(X, Y)        # r1 = a r, l1 = b ph l, rho = r1 conj(l1)
    if rmw != 0.0
        fch = V2.fmin_sub[s] + c * (V2.fs / V2.M_chan / 1e6); λ = V2.C_MPS / (fch * 1e6); φ = 2 * rmw * λ^2; cs = cos(φ); sn = sin(φ)
    else
        cs = 1.0; sn = 0.0
    end
    Qnl = 2real(ρ); Unl = 2imag(ρ)
    pre = (cs * Qnl + sn * Unl, -sn * Qnl + cs * Unl, rr - ll)
    if any(!iszero, T.dR[s]) || any(!iszero, T.dL[s])
        dR = T.dR[s][c]; dL = T.dL[s][c]                                      # voltage transform r' = r - d_R l, l' = l - d_L r
        rr2 = rr + abs2(dR) * ll - 2real(conj(dR) * ρ)
        ll2 = ll + abs2(dL) * rr - 2real(dL * ρ)
        ρ2 = ρ - conj(dL) * rr - dR * ll + dR * conj(dL) * conj(ρ)
    else
        rr2 = rr; ll2 = ll; ρ2 = ρ
    end
    Ic = rr2 + ll2; Vc = rr2 - ll2; Qc = 2real(ρ2); Uc = 2imag(ρ2)
    Vcb = (any(!iszero, T.mQV[s]) || any(!iszero, T.mUV[s])) ? Vc - (T.mQV[s][c] * Qc + T.mUV[s][c] * Uc) : Vc
    full = (Ic, cs * Qc + sn * Uc, -sn * Qc + cs * Uc, Vcb)
    return full, (Ic, Qc, Uc, Vc), pre
end
"matrices (k, i, c, s): Mf fully calibrated, Mp before the ellipticity and RM, Mn before the D-terms, per channel, not /32"
function chanmats(T::Tables, ig::Float64; rmw::Float64=T.RM)
    Mf = zeros(4, 4, NCH, NSB); Mp = zeros(4, 4, NCH, NSB); Mn = zeros(3, 4, NCH, NSB)
    E = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    for s = 1:NSB, c = 1:NCH, i = 1:4
        f, p, n = calib_point(T, s, c, ig, E[i]...; rmw=rmw)
        for k = 1:4; Mf[k, i, c, s] = f[k]; Mp[k, i, c, s] = p[k]; end
        for k = 1:3; Mn[k, i, c, s] = n[k]; end
    end
    Mf, Mp, Mn
end
"band value (sum over channels and subbands, /32 per subband as in the kernel) of M (k,i,c,s) on X (i,c,s)"
band(M, X) = [sum(M[k, i, c, s] * X[i, c, s] for i = 1:4, c = 1:NCH, s = 1:NSB) / NCH for k = 1:size(M, 1)]
"per-channel values (k, 128 = 32(s-1)+c) of M on X (i,c,s)"
function perchan2(M, X)
    out = zeros(size(M, 1), NCH * NSB)
    for s = 1:NSB, c = 1:NCH, k = 1:size(M, 1)
        out[k, NCH*(s-1)+c] = M[k, 1, c, s] * X[1, c, s] + M[k, 2, c, s] * X[2, c, s] + M[k, 3, c, s] * X[3, c, s] + M[k, 4, c, s] * X[4, c, s]
    end
    out
end
"band profile (k, t) over a span array Xs (i, c, s, t)"
function bandspan(M, Xs)
    n = size(Xs, 4); out = zeros(size(M, 1), n)
    @inbounds for t = 1:n, s = 1:NSB, c = 1:NCH, i = 1:4, k = 1:size(M, 1)
        out[k, t] += M[k, i, c, s] * Xs[i, c, s, t]
    end
    out ./ NCH
end
"per-channel calibrated span (k, 128, t)"
function chanspan(M, Xs)
    n = size(Xs, 4); out = zeros(size(M, 1), NCH * NSB, n)
    @inbounds for t = 1:n, s = 1:NSB, c = 1:NCH, k = 1:size(M, 1)
        out[k, NCH*(s-1)+c, t] = M[k, 1, c, s] * Xs[1, c, s, t] + M[k, 2, c, s] * Xs[2, c, s, t] + M[k, 3, c, s] * Xs[3, c, s, t] + M[k, 4, c, s] * Xs[4, c, s, t]
    end
    out
end
"per-channel per-sample baseline sigma of M's rows from the centred 4x4 moments H (i, j, c, s)"
function chansig(M, H, nb)
    out = zeros(size(M, 1), NCH * NSB)
    for s = 1:NSB, c = 1:NCH, k = 1:size(M, 1)
        v = 0.0
        for i = 1:4, j = 1:4; v += M[k, i, c, s] * H[i, j, c, s] * M[k, j, c, s]; end
        out[k, NCH*(s-1)+c] = sqrt(max(0.0, v / (nb - 1)))
    end
    out
end

# ------------------------------------------------------------------------------------------------ S/N of the calibrated band I
const T0OUT_ = V2.t_start_s + V2.edge_crop_ms * 1e-3
const PKRNG_ = (ceil(Int, (V2.PK_LO - T0OUT_) / 1e-6 + 0.5)):(floor(Int, (V2.PK_HI - T0OUT_) / 1e-6 + 0.5))
"""
    snr10_run(I, mu, sigma) -> (snr_window, snr_pk, pos_window, pos_pk)
10-us running-boxcar S/N of the calibrated band-I series: z[i] = (mean(I[i:i+9]) - mu) / (sigma / sqrt(10)), with mu and sigma
the mean and per-sample standard deviation of the first-20 % baseline. sigma/sqrt(10) assumes independent 1-us samples.
Returns the maximum over the window and over runs inside the peak window PK_LO..PK_HI, with the bin of each run's first sample.
"""
function snr10_run(I::Vector{Float64}, mu::Float64, sigma::Float64)
    n = length(I); cs = cumsum(vcat(0.0, I)); m = n - 9
    z = [((cs[i+10] - cs[i]) / 10 - mu) / (sigma / sqrt(10.0)) for i in 1:m]
    r = intersect(PKRNG_, 1:m); kw = argmax(z); kp = first(r) - 1 + argmax(@view z[r])
    (z[kw], z[kp], kw, kp)
end

# ------------------------------------------------------------------------------------------------ one catalogue window
readall(h, g) = Dict(k => read(h["$g/$k"]) for k in keys(h[g]))
"""
    calibrate_window(path, T, Trt) -> (prof, row, freq, extra, diag)
Calibrate one stored window with table set T (Trt = the tables the window was read with). The band profile inside the stored
span is the read-time profile plus the change of calibration applied to the stored coherency, which is exact for any table
set. Apertures, widths and noise are then measured again with the kernel's own measurement code.
"""
function calibrate_window(path::String, T::Tables, Trt::Tables)
    h = h5open(path, "r")
    sc = readall(h, "scalars"); pr = readall(h, "profiles")
    RC = Dict(k => read(h[k]) for k in ("RC_on", "RC_on_union", "RC_off", "RC_base", "RC_tot", "RC_mask", "RC_rawwin", "H_base", "G_base"))
    Xs = Float64.(read(h["RC_span"]))
    nseq = parse(Int, read(h["str/nseq"])); close(h)
    sod = sc["sod"]; phase = sc["phase"]; detsnr = sc["detsnr"]; nt = Int(sc["nt"]); nb = Int(sc["nb"])
    ig_rt = sc["inv_gamma"]; ig_rt == 1.0 / diffgain(Trt, sod) || error("$path: stored inverse gamma differs from the read-time table")
    ig = 1.0 / diffgain(T, sod)
    rmw = rm_total(T, sod)
    Mf, Mp, Mn = chanmats(T, ig; rmw=rmw); Mfr, _, _ = chanmats(Trt, ig_rt)
    Mfr = wscale(Mfr, Trt.chw)
    Mfw = wscale(Mf, T.chw); Mnw = wscale(Mn, T.chw)                  # band sums use the channel weights, per-channel outputs do not
    dM = Mfw .- Mfr
    slo = Int(sc["span_lo"]); shi = Int(sc["span_hi"]); ipk0 = Int(sc["peak_bin"]); R = V2.RAW_HALFWIN
    # first-20 % statistics: mu exact from the stored baseline mean, sigma^2 = [(B + 2 dB) . G + dB C dB] / (nb - 1) with the
    # baseline covariance C kept in per-channel 4x4 blocks
    μ = band(Mfw, RC["RC_base"]); G = RC["G_base"]; H = RC["H_base"]
    var = zeros(4)
    for k = 1:4
        v = 0.0
        for s = 1:NSB, c = 1:NCH
            for i = 1:4; v += ((Mfr[k, i, c, s] + 2dM[k, i, c, s]) / NCH) * G[k, i, c, s]; end
            for i = 1:4, j = 1:4; v += (dM[k, i, c, s] / NCH) * H[i, j, c, s] * (dM[k, j, c, s] / NCH); end
        end
        var[k] = v / (nb - 1)
    end
    σ = sqrt.(max.(var, 0.0))
    stats = (nb, μ[1], σ[1], μ[2], μ[3], μ[4], 0.5 * (σ[2] + σ[3]), σ[4])
    # full-length band profiles: inside the stored span read-time profile + change, outside it read-time profile + mean shift
    span_dif = bandspan(dM, Xs); span_abs = bandspan(Mfw, Xs)
    rawprof = (pr["I"], pr["Q"], pr["U"], pr["V"])
    shift = band(dM, RC["RC_off"])
    hyb = [fill(μ[k], nt) for k = 1:4]
    for b = 1:nt
        j = R + 1 + b - ipk0
        inraw = 1 <= j <= 2R + 1 && isfinite(rawprof[1][j])
        if slo <= b <= shi
            for k = 1:4; hyb[k][b] = inraw ? rawprof[k][j] + span_dif[k, b-slo+1] : span_abs[k, b-slo+1]; end
        elseif inraw
            for k = 1:4; hyb[k][b] = rawprof[k][j] + shift[k]; end
        end
    end
    δt = V2.TS_NAVG / V2.fs; t0 = V2.t_start_s + V2.edge_crop_ms * 1e-3; tax = t0 .+ ((1:nt) .- 0.5) .* δt
    diag = Dict{String,Any}("span_ok" => true, "ap3_changed" => false, "union_changed" => false)
    insp(a, b) = slo <= a && b <= shi
    xoff = Ref(RC["RC_off"])
    # off-pulse (nebula) coherency for the mask around the pulse: the stored one if the mask is unchanged, else rebuilt from the span
    function nebfn(ilo, ihi, offmask)
        gpad = max(3, ihi - ilo + 1); mlo = max(1, ilo - gpad); mhi = min(nt, ihi + gpad)
        if mlo == Int(sc["mask_lo"]) && mhi == Int(sc["mask_hi"])
            x = RC["RC_off"]
        else
            insp(min(mlo, Int(sc["mask_lo"])), max(mhi, Int(sc["mask_hi"]))) || (diag["span_ok"] = false)
            m = copy(RC["RC_mask"])
            for b = Int(sc["mask_lo"]):Int(sc["mask_hi"]); (mlo <= b <= mhi) || !(slo <= b <= shi) || (m .-= @view Xs[:, :, :, b-slo+1]); end
            for b = mlo:mhi; (Int(sc["mask_lo"]) <= b <= Int(sc["mask_hi"])) || !(slo <= b <= shi) || (m .+= @view Xs[:, :, :, b-slo+1]); end
            x = (RC["RC_tot"] .- m) ./ count(offmask)
        end
        xoff[] = x
        f = band(Mfw, x); n = band(Mnw, x)
        (f[1], f[2], f[3], f[4], n[1], n[2], n[3])
    end
    prof, row = V2.measure_from_profiles(hyb[1], hyb[2], hyb[3], hyb[4], tax, zeros(nt), zeros(nt), zeros(nt),
                                         nseq, sod, phase, detsnr, OBSDATE; stats=stats, nebfn=nebfn)
    chi = row.q_deg + T.bwg_alpha * row.el_deg + T.bwg_gamma * row.az_deg
    row = merge(row, (chi_bwg=chi,))
    ilo = prof.ap3_lo; ihi = prof.ap3_hi; ulo = prof.union_lo; uhi = prof.union_hi
    diag["ap3_changed"] = (ilo, ihi) != (Int(sc["ap3_lo"]), Int(sc["ap3_hi"]))
    diag["union_changed"] = (ulo, uhi) != (Int(sc["union_lo"]), Int(sc["union_hi"]))
    diag["peak_changed"] = prof.peak_bin != ipk0
    (insp(ilo, ihi) && insp(ulo, uhi)) || (diag["span_ok"] = false)
    snr10 = snr10_run(hyb[1], μ[1], σ[1])
    diag["dyn_ok"] = insp(ulo - 50, uhi + 50) || (ulo - 50 < 1 && slo == 1) || (uhi + 50 > nt && shi == nt)
    # aperture sums: the stored float64 sum when the aperture is unchanged, else the stored sum plus or minus the span samples
    # at its edges (the baseline term stays exact)
    function apd(lo, hi, lo0, hi0, X0)
        X = X0 .+ (hi0 - lo0 + 1) .* RC["RC_base"]
        for b = lo:hi; (lo0 <= b <= hi0) || (X .+= @view Xs[:, :, :, b-slo+1]); end
        for b = lo0:hi0; (lo <= b <= hi) || (X .-= @view Xs[:, :, :, b-slo+1]); end
        X .- (hi - lo + 1) .* RC["RC_base"]
    end
    Xon = diag["ap3_changed"] ? apd(ilo, ihi, Int(sc["ap3_lo"]), Int(sc["ap3_hi"]), RC["RC_on"]) : RC["RC_on"]
    Xun = diag["union_changed"] ? apd(ulo, uhi, Int(sc["union_lo"]), Int(sc["union_hi"]), RC["RC_on_union"]) : RC["RC_on_union"]
    fp_on = perchan2(Mp, Xon); fp_un = perchan2(Mp, Xun); fp_off = perchan2(Mp, xoff[])
    freq = (I_on=fp_on[1, :], Q_on=fp_on[2, :], U_on=fp_on[3, :], V_on=fp_on[4, :],
            I_off=fp_off[1, :], Q_off=fp_off[2, :], U_off=fp_off[3, :], V_off=fp_off[4, :],
            I_on_union=fp_un[1, :], Q_on_union=fp_un[2, :], U_on_union=fp_un[3, :], V_on_union=fp_un[4, :])
    Mf1, _, _ = chanmats(T, 1.0; rmw=rmw); fb1 = band(wscale(Mf1, T.chw), xoff[])      # nebular band Stokes with gamma(t) = 1
    sigR = sqrt(max(0.0, sum(H[1, 1, c, s] for c = 1:NCH, s = 1:NSB) / (nb - 1))) / NCH; sigL = sqrt(max(0.0, sum(H[2, 2, c, s] for c = 1:NCH, s = 1:NSB) / (nb - 1))) / NCH
    ff_on = perchan2(Mf, Xon); ff_un = perchan2(Mf, Xun); ff_off = perchan2(Mf, xoff[]); ff_base = perchan2(Mf, RC["RC_base"])
    sig_f = chansig(Mf, H, nb); sig_p = chansig(Mp, H, nb)
    dlo = max(slo, ulo - 50); dhi = min(shi, uhi + 50)
    dyn = chanspan(Mf, @view Xs[:, :, :, dlo-slo+1:dhi-slo+1])
    extra = (fcal_on=ff_on, fcal_union=ff_un, fcal_off=ff_off, fcal_base=ff_base, sig_full=sig_f, sig_pre=sig_p,
             RC_on=Xon, RC_on_union=Xun, RC_off=xoff[], RC_base=RC["RC_base"], chi=chi, ig=ig,
             dyn=dyn, dyn_lo=dlo, dyn_hi=dhi, ap3=(ilo, ihi), union=(ulo, uhi), W=(ihi - ilo + 1, uhi - ulo + 1), snr10=snr10, rm_ion=rm_ion_at(T, sod), rmw=rmw,
             sigq=σ[2], sigu=σ[3], nb=nb, sigR=sigR, sigL=sigL, VI_neb_preGamma=fb1[4] / fb1[1])
    return prof, row, freq, extra, diag
end

"""
    read_dynspec(item_path, T, Trt) -> (dyn, first_bin, ap3, union)
Calibrated dynamic spectrum of one stored window: dyn is (4 = I,Q,U,V; 128 channels subband-major; n_us), feed frame, not /32,
over the union aperture +- 50 us. first_bin is the window bin of its first sample; ap3 and union are the apertures measured with T.
"""
function read_dynspec(path::String, T::Tables, Trt::Tables)
    _, _, _, ex, _ = calibrate_window(path, T, Trt)
    return ex.dyn, ex.dyn_lo, ex.ap3, ex.union
end
