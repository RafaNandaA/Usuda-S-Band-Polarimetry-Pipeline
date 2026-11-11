# Off-pulse nebula snapshots, the calibrator of every polarization term. Included by extract/extract.jl after the kernel
# module K is loaded (it uses K's constants, the R/L delay ramp and the geometry functions).
#
# The Crab Nebula fills the beam and has a steady linear polarization, and the track has no calibrator scan, so the
# off-pulse sky is the only calibrator. Each snapshot is 10^7 words (156 ms per subband) at rotation phase 0.7, between
# the main pulse and the interpulse. The snapshots are spread evenly in rotation number over the track.
# Listed giant pulses (both lists, all rows) that fall in a snapshot are masked along their dispersed track in each
# subband, +- mask_pad_ms, in 64-sample blocks, because a bright pulse adds polarized power the nebula does not have.
#
# Processing per subband and hand: real samples -> long FFT -> analytic-signal operator (sqrt2 on bins 2..N, 1 at the
# Nyquist bin, 0 elsewhere, no dispersion: the nebula is steady) and for L also the R/L delay ramp -> inverse FFT -> crop
# stage1_crop samples at each end (the ramp's circular shift wraps there) -> 64-point channelisation, channels 1..32 =
# FFT bins 2..33 as in the pulse kernel. Coherency (RR, LL, RL) is averaged over all blocks (unmasked) and over the kept
# blocks (masked). No other calibration is applied: the fit derives everything from these snapshots.
#
# A second series of 1-s snapshots around the R-chain gain step (no listed pulse falls there) measures the step amplitude.
#
# Outputs (h5py view):
#   nebula_file     coherency/rho_RR, rho_LL, rho_RL_re, rho_RL_im (K, 4, 32), masked, coherency_unmasked/ the same with
#                   all blocks, geometry/toff_sod, q_deg, el_deg, az_deg, chi_BWG_deg, I_band (K,), frequency/freq_MHz
#                   (4, 32), fmin_sub_MHz, mask/masked_fraction, nkeep (K, 4), nblk, n_listed_grp (K,),
#                   monitors/P, frac_level0, frac_level15 (K, 158, 8), bin_1ms, n (K, 158)
#   gain_step_file  coherency/* (80, 4, 32), file_time_start_s, nblk, frequency/freq_MHz, monitors/*

const NB_CH = K.NCHAN; const NSB_N = K.n_subbands
const NCFG = CFG["nebula"]
const SOD_FILE = 28800.0                 # file start, 08:00:00 UTC (file name), file time = SOD_FILE + word / fs
const NEB_DIR = joinpath(CFG["paths"]["work_dir"], "nebula")
const NEB_ITEMS = joinpath(CFG["paths"]["work_dir"], "nebula_items.tsv")
const NEB_OBSDATE = Date(string(CFG["observation"]["date"]))

# ------------------------------------------------------------------------------------------------ plan
"all rows of both detection lists (no S/N cut): Nseq, UTC list time, phase"
function read_all_list_rows(files)
    nseqs = Int[]; dett = Float64[]; phs = Float64[]
    for f in sort(files), ln in eachline(f)
        s = strip(ln)
        (isempty(s) || startswith(s, "#")) && continue
        t = split(s); length(t) < 5 && continue
        push!(dett, K.parse_float_safe(t[2])); push!(nseqs, round(Int, K.parse_float_safe(t[4]))); push!(phs, K.parse_float_safe(t[5]))
    end
    nseqs, dett, phs
end

"listed-pulse mask of a read (w0, nw words): per subband, word intervals of the dispersed tracks +- pad"
function snapshot_masks(w0, nw, ts, nq, pad)
    a = SOD_FILE + w0 / K.fs; b = SOD_FILE + (w0 + nw) / K.fs
    j0 = searchsortedfirst(ts, a - 0.01); j1 = searchsortedfirst(ts, b + 0.001)
    dt(f) = 4.148808e3 * K.DM * (1.0 / f^2 - 1.0 / K.f_ref_MHz^2)
    out = [Tuple{Int,Int}[] for _ = 1:4]; grp = Set{Int}()
    for j = j0:j1-1, s = 1:4
        lo = ts[j] + dt(K.fmin_sub[s] + 32.0) - pad; hi = ts[j] + dt(K.fmin_sub[s]) + pad
        wl = floor(Int, (lo - SOD_FILE) * K.fs); wh = ceil(Int, (hi - SOD_FILE) * K.fs)
        if wh > w0 && wl < w0 + nw
            push!(out[s], (max(wl, w0), min(wh, w0 + nw))); push!(grp, nq[j])
        end
    end
    out, sort(collect(grp))
end
mask_string(m) = join(["$(s-1):" * join(["$(a)-$(b)" for (a, b) in m[s]], ",") for s = 1:4 if !isempty(m[s])], "|")

function plan_nebula()
    nseqs, dett, phs = read_all_list_rows([CFG["paths"]["detlist_R"], CFG["paths"]["detlist_L"]])
    # rotation phase 0 time against Nseq (quadratic), then the snapshot at off_phase of the local period
    N = Float64.(nseqs)
    P0 = ([ones(length(N)) N] \ dett)[2]
    tf = dett .- phs .* P0
    c = [ones(length(N)) N N.^2] \ tf
    Kn = Int(CFG["calibration"]["n_snapshots"]); nw = Int(NCFG["snapshot_words"])
    Ns = round.(Int, range(minimum(nseqs), maximum(nseqs), length=Kn))
    toff = [c[1] + c[2]*n + c[3]*n^2 + Float64(NCFG["off_phase"]) * (c[2] + 2*c[3]*n) for n in Ns]
    # 2322-MHz arrival of every listed pulse in file time: list time + 31.4 ms (file clock) + arrival offset in the read
    C0 = Float64(CFG["timing"]["arrival_offset_ms"]); pad = Float64(NCFG["mask_pad_ms"]) * 1e-3
    t2322 = dett .+ (SOD_FILE - K.SOD_RAW_START) .+ C0 * 1e-3
    o = sortperm(t2322); ts = t2322[o]; nq = nseqs[o]
    rows = String[]
    for k = 1:Kn
        w0 = round(Int, (toff[k] - SOD_FILE) * K.fs); m, g = snapshot_masks(w0, nw, ts, nq, pad)
        push!(rows, join(("neb:$(k-1)", "nebula", w0 * 4, nw * 4, -1,
                          "snap=$(k-1);toff=$(toff[k]);grp=$(join(g, ","));mask=$(mask_string(m))"), '\t'))
    end
    s0 = Int(NCFG["gain_step_start_s"])
    for sec = s0:s0+Int(NCFG["gain_step_n"])-1
        w0 = round(Int, (sec - SOD_FILE) * K.fs); m, g = snapshot_masks(w0, 64_000_000, ts, nq, pad)
        push!(rows, join(("gstep:$sec", "gainstep", w0 * 4, 64_000_000 * 4, -1,
                          "sec=$sec;grp=$(join(g, ","));mask=$(mask_string(m))"), '\t'))
    end
    mkpath(dirname(NEB_ITEMS))
    open(NEB_ITEMS, "w") do io
        println(io, "item\tkind\toffset_bytes\tlength_bytes\tblock\tmeta"); foreach(r -> println(io, r), rows)
    end
    @printf("%s: %d nebula snapshots, %d gain-step seconds; %d snapshots hold a listed pulse\n", NEB_ITEMS, Kn,
            Int(NCFG["gain_step_n"]), count(r -> !occursin("grp=;", r), rows[1:Kn]))
end

# ------------------------------------------------------------------------------------------------ engine
function hilbert_op(Lwin::Int)
    h = zeros(ComplexF64, Lwin); N = Lwin ÷ 2
    h[2:N] .= sqrt(2.0); h[N+1] = 1.0; h
end
const _NLOCK = ReentrantLock()
const _S1PLANS = Dict{Int,Any}()
"FFT plans (long forward, long inverse, 64-point along dim 1) and the operators H (R) and H x ramp (L, per subband)"
function s1plans(seg::Int, crop::Int)
    lock(_NLOCK) do
        haskey(_S1PLANS, seg) && return _S1PLANS[seg]
        nb = (seg - 2crop) ÷ K.M_chan
        p = (plan_fft(zeros(ComplexF64, seg)), plan_ifft(zeros(ComplexF64, seg)),
             plan_fft(zeros(ComplexF64, K.M_chan, nb), 1), hilbert_op(seg),
             Dict(s => hilbert_op(seg) .* K.ramp_L(seg, K.fmin_sub[s], K.delay_RL[s]) for s = 1:NSB_N))
        _S1PLANS[seg] = p; return p
    end
end
struct DumpWS
    x::Vector{ComplexF64}; X::Vector{ComplexF64}; CRb::Matrix{ComplexF64}
end
DumpWS(seg::Int, NB::Int) = DumpWS(Vector{ComplexF64}(undef, seg), Vector{ComplexF64}(undef, seg), Matrix{ComplexF64}(undef, K.M_chan, NB))
const _DWSPOOL = Dict{Tuple{Int,Int},Vector{DumpWS}}()
function with_dws(f, seg::Int, NB::Int)
    ws = lock(_NLOCK) do
        v = get!(() -> DumpWS[], _DWSPOOL, (seg, NB)); isempty(v) ? DumpWS(seg, NB) : pop!(v)
    end
    try; return f(ws); finally; lock(() -> push!(_DWSPOOL[(seg, NB)], ws), _NLOCK); end
end
mutable struct Acc4; RR::Matrix{Float64}; LL::Matrix{Float64}; RL::Matrix{ComplexF64}; end
Acc4() = Acc4(zeros(NSB_N, NB_CH), zeros(NSB_N, NB_CH), zeros(ComplexF64, NSB_N, NB_CH))
"sum over all blocks (F) and over the kept blocks (M) of one segment, per channel in block order"
function accum!(F::Acc4, M::Acc4, CR, CL, kb, s::Int, NB::Int)
    @inbounds for c = 1:NB_CH
        sRR = 0.0; sLL = 0.0; sRL = zero(ComplexF64); mRR = 0.0; mLL = 0.0; mRL = zero(ComplexF64)
        for b = 1:NB
            r = CR[c+1, b]; l = CL[c+1, b]
            sRR += abs2(r); sLL += abs2(l); sRL += r*conj(l)
            if kb[b]; mRR += abs2(r); mLL += abs2(l); mRL += r*conj(l); end
        end
        F.RR[s,c] += sRR; F.LL[s,c] += sLL; F.RL[s,c] += sRL
        M.RR[s,c] += mRR; M.LL[s,c] += mLL; M.RL[s,c] += mRL
    end
end
function subband_sums!(Bf, Bm, buf, s, nseg, seg, crop, NB, FP, IP, CP, H, HLs, ws, keep::BitVector)
    x = ws.x; X = ws.X; CR = ws.CRb; Mc = size(CR, 1)
    GC.@preserve x X begin
        zin = unsafe_wrap(Array, pointer(x, crop+1), (Mc, NB))   # the cropped analytic signal as M_chan x NB blocks
        CL  = unsafe_wrap(Array, pointer(X, 1), (Mc, NB))        # X's memory reused for the L channel matrix
        for w = 0:nseg-1
            o = w*seg; kb = view(keep, w*NB+1:(w+1)*NB)
            @inbounds for n = 1:seg; x[n] = K.extract(buf[o+n], s-1); end
            mul!(X, FP, x); X .*= H; mul!(x, IP, X)
            mul!(CR, CP, zin)
            @inbounds for n = 1:seg; x[n] = K.extract(buf[o+n], s-1+4); end
            mul!(X, FP, x); X .*= HLs; mul!(x, IP, X)
            mul!(CL, CP, zin)
            accum!(Bf, Bm, CR, CL, kb, s, NB)
        end
    end
    nothing
end
"keep[b] for every 64-sample block of every segment of one subband, from absolute word intervals"
function keepmask(w0::Int, nseg::Int, seg::Int, crop::Int, NB::Int, iv::Vector{Tuple{Int,Int}})
    k = trues(nseg * NB)
    for (lo, hi) in iv, w = 0:nseg-1
        base = w0 + w*seg + crop                      # absolute word of block 1's first sample in segment w
        b1 = max(1, fld(lo - base, 64) + 1); b2 = min(NB, cld(hi - base, 64))
        b1 <= b2 && (k[w*NB .+ (b1:b2)] .= false)
    end
    k
end
function parse_mask(s::AbstractString)
    m = [Tuple{Int,Int}[] for _ = 1:4]
    isempty(s) && return m
    for part in split(s, '|')
        sb, rest = split(part, ':'); for iv in split(rest, ','); a, b = split(iv, '-'); push!(m[parse(Int, sb)+1], (parse(Int, a), parse(Int, b))); end
    end
    m
end
"mean coherency over all blocks (Bf) and over the unmasked blocks (Bm) of one read"
function snapshot_sums(buf::Vector{UInt32}, w0::Int, maskiv; seg::Int, crop::Int)
    nseg = length(buf) ÷ seg
    FP, IP, CP, H, HL = s1plans(seg, crop)
    NB = (seg - 2crop) ÷ K.M_chan
    Bf = Acc4(); Bm = Acc4()
    keeps = [keepmask(w0, nseg, seg, crop, NB, maskiv[s]) for s = 1:4]
    @sync for s = 1:NSB_N
        Threads.@spawn with_dws(ws -> subband_sums!(Bf, Bm, buf, s, nseg, seg, crop, NB, FP, IP, CP, H, HL[s], ws, keeps[s]), seg, NB)
    end
    nblk = nseg * NB; nkeep = [count(keeps[s]) for s = 1:4]
    Bf.RR ./= nblk; Bf.LL ./= nblk; Bf.RL ./= nblk
    for s = 1:4; Bm.RR[s, :] ./= nkeep[s]; Bm.LL[s, :] ./= nkeep[s]; Bm.RL[s, :] ./= nkeep[s]; end
    return Bf, Bm, nblk, nkeep
end
wacc(h, g, A) = (h["$g/RR"] = A.RR; h["$g/LL"] = A.LL; h["$g/RLre"] = real.(A.RL); h["$g/RLim"] = imag.(A.RL))

# ------------------------------------------------------------------------------------------------ run
function run_nebula()
    isfile(NEB_ITEMS) || error("$NEB_ITEMS not found: run the plan first")
    mkpath(joinpath(NEB_DIR, "items"))
    donef = joinpath(NEB_DIR, "done.list"); done = Set(isfile(donef) ? filter(!isempty, readlines(donef)) : String[])
    its = [split(l, '\t') for l in readlines(NEB_ITEMS)[2:end]]
    MAXITEMS > 0 && (its = its[1:min(end, MAXITEMS)])
    srcpath = SOURCE == :disk ? RAW : SLICES
    isfile(srcpath) || error("input not found: $srcpath")
    crop = Int(NCFG["stage1_crop"]); prov = K.provenance_string() * "; nebula snapshots at DM 0, crop $crop, listed-pulse mask"
    io = open(srcpath, "r"); t0 = time(); n = 0
    try
        for t in its
            id = String(t[1]); id in done && continue
            off = parse(Int, t[3]); len = parse(Int, t[4]); meta = parse_meta(t[6])
            buf = Vector{UInt32}(undef, len ÷ 4); read_range!(io, buf, srcoff(off, len))
            mon = monitors(buf, off ÷ 4)
            seg = t[2] == "nebula" ? Int(NCFG["snapshot_words"]) : Int(NCFG["gain_step_segment_words"])
            Bf, Bm, nblk, nkeep = snapshot_sums(buf, off ÷ 4, parse_mask(get(meta, "mask", "")); seg=seg, crop=crop)
            (all(isfinite, Bf.RL) && all(isfinite, Bm.RL)) || @warn "non-finite output" id
            h5open(joinpath(NEB_DIR, "items", safe(id) * ".h5"), "w") do h
                wacc(h, "B_full", Bf); wacc(h, "B_masked", Bm)
                h["nblk"] = nblk; h["nkeep"] = nkeep; h["masked_fraction"] = 1 .- nkeep ./ nblk
                h["n_grp"] = isempty(get(meta, "grp", "")) ? 0 : length(split(meta["grp"], ','))
                write_mon(h, mon); attributes(h)["provenance"] = prov; attributes(h)["item"] = id
            end
            open(io2 -> println(io2, id), donef, "a"); n += 1
            n % 50 == 0 && @printf("%d items, %.0f s\n", n, time() - t0)
        end
    finally
        close(io)
    end
    @printf("nebula items done: %d read in %.0f s\n", n, time() - t0)
    MAXITEMS > 0 || assemble_nebula(its, prov)
end

g4(h, n) = (read(h["$n/RR"]), read(h["$n/LL"]), complex.(read(h["$n/RLre"]), read(h["$n/RLim"])))
pk(A) = permutedims(A, (2, 1, 3))                 # (4,32,K) -> (32,4,K) in Julia = (K,4,32) in h5py
function assemble_nebula(its, prov)
    neb = [t for t in its if t[2] == "nebula"]; gst = [t for t in its if t[2] == "gainstep"]
    Kn = length(neb); fr = [K.fmin_sub[s] + c * (K.fs / K.M_chan / 1e6) for c = 1:NB_CH, s = 1:NSB_N]   # h5py (4, 32)
    RRm = zeros(4, 32, Kn); LLm = zeros(4, 32, Kn); RLm = zeros(ComplexF64, 4, 32, Kn)
    RRf = zeros(4, 32, Kn); LLf = zeros(4, 32, Kn); RLf = zeros(ComplexF64, 4, 32, Kn)
    mf = zeros(4, Kn); nkeep = zeros(Int, 4, Kn); nblk = zeros(Int, Kn); ngrp = zeros(Int, Kn)
    MB = 158; monP = fill(NaN32, 8, MB, Kn); mon0 = fill(NaN32, 8, MB, Kn); mon15 = fill(NaN32, 8, MB, Kn); monbin = fill(-1, MB, Kn); monn = zeros(Int32, MB, Kn)
    toff = zeros(Kn); q = zeros(Kn); el = zeros(Kn); az = zeros(Kn); chi = zeros(Kn)
    for (i, t) in enumerate(neb)
        h5open(joinpath(NEB_DIR, "items", safe(String(t[1])) * ".h5"), "r") do h
            a = g4(h, "B_masked"); RRm[:, :, i] = a[1]; LLm[:, :, i] = a[2]; RLm[:, :, i] = a[3]
            a = g4(h, "B_full"); RRf[:, :, i] = a[1]; LLf[:, :, i] = a[2]; RLf[:, :, i] = a[3]
            mf[:, i] = read(h["masked_fraction"]); nkeep[:, i] = read(h["nkeep"]); nblk[i] = read(h["nblk"]); ngrp[i] = read(h["n_grp"])
            nb = length(read(h["mon/bin"]))
            monP[:, 1:nb, i] = read(h["mon/P"]); mon0[:, 1:nb, i] = read(h["mon/f0"]); mon15[:, 1:nb, i] = read(h["mon/f15"])
            monbin[1:nb, i] = read(h["mon/bin"]); monn[1:nb, i] = read(h["mon/n"])
        end
        toff[i] = parse(Float64, parse_meta(t[6])["toff"])
        u = DateTime(NEB_OBSDATE) + Millisecond(round(Int, toff[i] * 1000))
        q[i] = K.parallactic_deg(u); el[i] = K.elevation_deg(u); az[i] = K.azimuth_deg(u)
        chi[i] = q[i] + K.BWG_ALPHA * el[i] + K.BWG_GAMMA * az[i]
    end
    pf = CFG["paths"]["nebula_file"]; mkpath(dirname(pf))
    h5open(pf * ".part", "w") do h
        h["coherency/rho_RR"] = pk(RRm); h["coherency/rho_LL"] = pk(LLm)
        h["coherency/rho_RL_re"] = pk(real.(RLm)); h["coherency/rho_RL_im"] = pk(imag.(RLm))
        h["coherency_unmasked/rho_RR"] = pk(RRf); h["coherency_unmasked/rho_LL"] = pk(LLf)
        h["coherency_unmasked/rho_RL_re"] = pk(real.(RLf)); h["coherency_unmasked/rho_RL_im"] = pk(imag.(RLf))
        h["geometry/toff_sod"] = toff; h["geometry/q_deg"] = q; h["geometry/el_deg"] = el; h["geometry/az_deg"] = az
        h["geometry/chi_BWG_deg"] = chi; h["geometry/I_band"] = [sum(RRm[:, :, i]) + sum(LLm[:, :, i]) for i = 1:Kn]
        h["frequency/freq_MHz"] = fr; h["frequency/fmin_sub_MHz"] = collect(K.fmin_sub)
        h["mask/masked_fraction"] = mf; h["mask/nkeep"] = nkeep; h["mask/nblk"] = nblk; h["mask/n_listed_grp"] = ngrp
        h["monitors/P"] = monP; h["monitors/frac_level0"] = mon0; h["monitors/frac_level15"] = mon15; h["monitors/bin_1ms"] = monbin; h["monitors/n"] = monn
        a = attributes(h)
        a["BWG_ALPHA"] = K.BWG_ALPHA; a["BWG_GAMMA"] = K.BWG_GAMMA; a["CRAB_RA_DEG"] = K.CRAB_RA_DEG; a["CRAB_DEC_DEG"] = K.CRAB_DEC_DEG
        a["USUDA_LAT_DEG"] = K.USUDA_LAT_DEG; a["USUDA_LON_DEG"] = K.USUDA_LON_DEG; a["fs_Hz"] = Float64(K.fs)
        a["M_chan"] = Int64(K.M_chan); a["NCHAN"] = Int64(NB_CH); a["n_subbands"] = Int64(NSB_N); a["n_snapshots"] = Int64(Kn)
        a["OFF_PHASE"] = Float64(NCFG["off_phase"]); a["chunk_samples"] = Int64(NCFG["snapshot_words"]); a["obs_date"] = string(NEB_OBSDATE)
        a["raw_file"] = basename(RAW); a["leakage_applied"] = Int64(0); a["bandpass_applied"] = Int64(0); a["gain_applied"] = Int64(0)
        a["phase_applied"] = Int64(0); a["delay_applied"] = Int64(1); a["delay_RL_s"] = collect(K.delay_RL)
        a["coordinate_basis"] = "feed angle: chi = q + alpha*el + gamma*az"
        a["units_coherency"] = "analytic-signal channelised voltages (R/L delay ramp on L applied, no other calibration)"
        a["segment_words"] = Int64(NCFG["snapshot_words"]); a["crop"] = Int64(NCFG["stage1_crop"])
        a["mask"] = "listed pulses (both lists, all rows): per-subband dispersed track at the configured DM +- $(NCFG["mask_pad_ms"]) ms, " *
                    "removed in 64-sample blocks; coherency/ = masked, coherency_unmasked/ = all blocks"
        a["provenance"] = prov
    end
    mv(pf * ".part", pf; force=true)
    KR = length(gst); rRR = zeros(4, 32, KR); rLL = zeros(4, 32, KR); rRL = zeros(ComplexF64, 4, 32, KR); rn = zeros(Int, KR)
    rP = fill(NaN32, 8, 1001, KR); r0 = similar(rP); r15 = similar(rP); rb = fill(-1, 1001, KR); secs = zeros(KR)
    for (i, t) in enumerate(gst)
        h5open(joinpath(NEB_DIR, "items", safe(String(t[1])) * ".h5"), "r") do h
            a = g4(h, "B_full"); rRR[:, :, i] = a[1]; rLL[:, :, i] = a[2]; rRL[:, :, i] = a[3]; rn[i] = read(h["nblk"])
            nb = length(read(h["mon/bin"])); rP[:, 1:nb, i] = read(h["mon/P"]); r0[:, 1:nb, i] = read(h["mon/f0"]); r15[:, 1:nb, i] = read(h["mon/f15"]); rb[1:nb, i] = read(h["mon/bin"])
            all(==(0), read(h["masked_fraction"])) || @warn "gain-step second $(t[1]) holds a listed pulse (unmasked sums are stored)"
        end
        secs[i] = parse(Float64, parse_meta(t[6])["sec"])
    end
    rf = CFG["paths"]["gain_step_file"]
    h5open(rf * ".part", "w") do h
        h["coherency/rho_RR"] = pk(rRR); h["coherency/rho_LL"] = pk(rLL); h["coherency/rho_RL_re"] = pk(real.(rRL)); h["coherency/rho_RL_im"] = pk(imag.(rRL))
        h["file_time_start_s"] = secs; h["nblk"] = rn; h["frequency/freq_MHz"] = fr
        h["monitors/P"] = rP; h["monitors/frac_level0"] = r0; h["monitors/frac_level15"] = r15; h["monitors/bin_1ms"] = rb
        attributes(h)["provenance"] = prov; attributes(h)["segment_words"] = Int64(NCFG["gain_step_segment_words"]); attributes(h)["crop"] = Int64(NCFG["stage1_crop"])
        attributes(h)["note"] = "file time = 28800.0 + word/64e6; one 1-s snapshot per row, all blocks"
    end
    mv(rf * ".part", rf; force=true)
    @printf("wrote %s (%d snapshots) and %s (%d seconds)\n", pf, Kn, rf, KR)
end
