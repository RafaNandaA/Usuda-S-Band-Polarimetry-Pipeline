# Window reader: one pass over the raw file for the catalogue windows and the fold rotations of the item list.
# Included by extract/extract.jl after the kernel module K (fold_kernel.jl) is loaded. Read-only access to the raw
# file, and no raw bytes are written to disk.
#
# Design
#   reader   (interactive thread): merges overlapping items into reads and reads them in file-offset order in <= 16 MiB
#            pieces into RAM buffers, under a byte budget (--queue-mb). Every read is logged to reads.tsv.
#   workers  (default threads): 1-ms monitors per stream (R0-3, L0-3), then the item's engine.
#   outputs  one HDF5 per catalogue window (items/cat_N.h5), written to *.part and renamed, then the item id is appended
#            to done.list. Fold rotations go to a collector that holds a block until all its rotations are in, applies
#            the flag rule, sums in rotation order and writes blocks/block_B.h5. A run that stops is resumed by running
#            the same command again: done items and blocks are skipped.
#   HDF5 access is serialised by one lock (the library build is not thread-safe).
#
# Per catalogue window (items/cat_N.h5), all float64 unless stated:
#   scalars/*, profiles/{I,Q,U,V}   the window measured with the read-time tables (reference row, profiles in the
#                                   raw_profiles layout, +-750 us around the peak, band-summed, calibrated)
#   RC_base  mean over bins 1:nb of the uncalibrated per-channel coherency (RR, LL, Re RL, Im RL, 32 ch, 4 subbands)
#   RC_on, RC_on_union   aperture sums minus W x RC_base (3-sigma run, union)      RC_off  mean over the off-pulse mask
#   RC_tot   sum over all bins       RC_mask  sum over the masked pulse region [ilo-gpad, ihi+gpad]
#   RC_rawwin sum over the raw-profile window bins (peak +-750, inside the window)
#   H_base   per channel 4x4 centred second moments over bins 1:nb.  G_base (4 Stokes x 4 x 32 x 4) = sum_t (S_k(t)-mu_k)
#            (x(t)-xbar) over bins 1:nb, S_k = the read-time calibrated band profile, so the baseline noise can be
#            propagated to any later table set
#   RC_span  float32 per-sample coherency over [union_lo - span_pad, union_hi + span_pad] (clipped): the dynamic
#            spectrum, with a margin for apertures recomputed under the final tables
#   snr/*    per-hand (R, L) and Stokes-I S/N: uncalibrated power summed over 4 subbands x 32 ch (table independent) and
#            the read-time calibrated band I, at 1-us peak, 10-us running boxcar, 10-us blocks, baseline = first 20 %,
#            maximum over the whole window and over the peak window PK_LO..PK_HI, with positions
#   mon/*, flag   1-ms monitors (P, frac0, frac15 per stream) and flag bits (1 interference interval, 2 power, 4 levels,
#                 8 incomplete) against a window-internal reference. Windows are flagged, never dropped.
# Per fold block (blocks/block_B.h5): RC16 = sum over unflagged rotations of the uncalibrated per-channel coherency per
#   16-us bin of each 2-ms window (MP, OFF, LFIP, 'all' and 'free' = rotations without a listed pulse), float32,
#   the read-time calibrated 1-us profiles (reference), and per rotation the flags, labels, power, time, gamma and geometry.

const NCH = K.NCHAN; const NSB = K.n_subbands
const OBSDATE = Date(string(CFG["observation"]["date"]))
const SPAN_PAD = Int(cfgget(CFG, "aperture", "span_pad_us", 70))   # us beyond the union aperture on each side
const PROV_READ = K.provenance_string() * "; stored: uncalibrated per-channel coherency per window and per fold block, " *
                  "with reference rows calibrated by the read-time tables"

# ------------------------------------------------------------------------------------------------ items and reads
struct Item; id::String; kind::Symbol; off::Int; len::Int; block::Int; meta::Dict{String,String}; end
struct Rd; off::Int; len::Int; items::Vector{Item}; end
items = Item[]
for ln in readlines(ITEMS)[2:end]
    t = split(ln, '\t'); k = Symbol(t[2]); k in (:catwin, :foldrot) || continue
    push!(items, Item(t[1], k, parse(Int, t[3]), parse(Int, t[4]), parse(Int, t[5]), parse_meta(t[6])))
end
sort!(items, by = it -> (it.off, it.id))
MAXITEMS > 0 && (items = items[1:min(end, MAXITEMS)])
reads = Rd[]
for it in items
    if !isempty(reads) && it.off <= reads[end].off + reads[end].len
        r = reads[end]; e = max(r.off + r.len, it.off + it.len)
        reads[end] = Rd(r.off, e - r.off, vcat(r.items, it))
    else
        push!(reads, Rd(it.off, it.len, [it]))
    end
end
const BLOCK_N = Dict{Int,Int}()
for it in items; it.kind == :foldrot && (BLOCK_N[it.block] = get(BLOCK_N, it.block, 0) + 1); end
# catalogue items must be the kernel's own detection-list rows and read offsets
let (det, od) = K.load_detlist([CFG["paths"]["detlist_R"], CFG["paths"]["detlist_L"]]), nbad = 0
    od == OBSDATE || error("detection lists are dated $od, the configuration says $OBSDATE")
    for it in items
        it.kind == :catwin || continue
        d = det[parse(Int, it.meta["nseq"])]
        nbad += !(d.sod == parse(Float64, it.meta["sod"]) && d.phase == parse(Float64, it.meta["phase"]) &&
                  d.detsnr == parse(Float64, it.meta["detsnr"]))
        nbad += round(Int, (d.sod - K.T_READ_START + K.t_start_s - K.SOD_RAW_START) * K.fs) * 4 != it.off
    end
    nbad == 0 || error("$nbad catalogue items differ from the detection-list rows or offsets: rerun the plan")
end
mkpath(joinpath(ODIR, "items")); mkpath(joinpath(ODIR, "blocks"))
const DONEF = joinpath(ODIR, "done.list")
const DONE = Set{String}(isfile(DONEF) ? filter(!isempty, readlines(DONEF)) : String[])
isdone(it::Item) = it.kind == :foldrot ? ("block:$(it.block)" in DONE) : (it.id in DONE)
const H5LOCK = ReentrantLock(); const DLOCK = ReentrantLock(); const LOGLOCK = ReentrantLock()
function mark_done(id)
    lock(DLOCK) do
        open(DONEF, "a") do io; println(io, id); flush(io); end
        push!(DONE, id)
    end
end
logline(file, s) = lock(LOGLOCK) do; open(io -> (print(io, s); flush(io)), joinpath(ODIR, file), "a"); end
function h5write_atomic(path, f)
    lock(H5LOCK) do
        tmp = path * ".part"
        h5open(f, tmp, "w")
        mv(tmp, path; force=true)
    end
end

allfinite(x) = all(isfinite, x)
const nbad = Threads.Atomic{Int}(0)
function note_bad(id, msg)
    Threads.atomic_add!(nbad, 1)
    logline("bad.log", @sprintf("%s\t%s\t%s\n", Dates.format(now(), "HH:MM:SS"), id, msg))
end

# ------------------------------------------------------------------------------------------------ flag rule
# Flags are stored, never used to drop a window, so the analysis can test flagged against unflagged windows.
#   1  read overlaps the interval with impulsive interference in R subbands 0-1
#   2  1-ms power of a stream deviates from its median by more than 6 times max(sigma_P/P, 0.0056), outside the pulse.
#      0.0056 = 1/sqrt(32 MHz x 1 ms) is the radiometer fractional sigma of one subband in 1 ms
#   4  occupancy of the extreme levels (0 and 15) above max(5 x median, 0.005): digitiser overload
#   8  incomplete read or non-finite output
const KDM = 4.148808e3; const FSUB = K.fmin_sub
dtf(DM, f) = KDM * DM * (1 / f^2 - 1 / K.f_ref_MHz^2)
const F1LO, F1HI = Float64.(CFG["flags"]["interference_interval_s"])   # file time (28800.0 + word/fs)
"monitor bins (4 subbands x bins) touched by the dispersed pulse at dedispersed raw words sXs (DM bracket 56.65-56.80)"
function pulse_bins(mon, sXs)
    pulse = falses(4, length(mon.bin))
    for sx in sXs, s = 1:4
        tlo = sx - round(Int, (1.0e-3 - dtf(56.65, FSUB[s] + 32) + 0.1e-3) * 64e6)
        thi = sx + round(Int, (1.0e-3 + dtf(56.80, FSUB[s]) + 0.1e-3) * 64e6)
        for (b, bi) in enumerate(mon.bin); (bi * WPB < thi && (bi + 1) * WPB > tlo) && (pulse[s, b] = true); end
    end
    pulse
end
"flag bits for one read (w0, nw) with monitor m, pulse mask, and per-stream references Pm, Ps, Em"
function flag_one(w0, nw, m, pulse, Pm, Ps, Em, ok)
    fl = 0; maxdev = 0.0; maxe = 0.0
    ft0 = 28800.0 + w0 / 64e6; ft1 = 28800.0 + (w0 + nw) / 64e6
    (ft1 > F1LO && ft0 < F1HI) && (fl |= 1)
    nfull = 0
    for (b, bi) in enumerate(m.bin)
        m.n[b] == WPB || continue
        nfull += 1
        for st = 1:8
            s = (st - 1) % 4 + 1
            e = m.f0[st, b] + m.f15[st, b]; maxe = max(maxe, e / max(5Em[st], 0.005))
            e > max(5Em[st], 0.005) && (fl |= 4)
            if !pulse[s, b]
                dev = abs(m.P[st, b] / Pm[st] - 1) / (6 * max(Ps[st] / Pm[st], 0.0056))
                maxdev = max(maxdev, dev); dev > 1 && (fl |= 2)
            end
        end
    end
    (!ok || nfull < fld(nw, WPB) - 1) && (fl |= 8)
    return fl, maxdev, maxe
end
"per-stream median power, robust sigma of the power, median extreme-level occupancy over the full 1-ms bins"
function refs(mons)
    allP = [Float64[] for _ = 1:8]; allE = [Float64[] for _ = 1:8]
    for m in mons, b in eachindex(m.bin)
        m.n[b] == WPB || continue
        for s = 1:8; push!(allP[s], m.P[s, b]); push!(allE[s], m.f0[s, b] + m.f15[s, b]); end
    end
    Pm = [median(x) for x in allP]; Ps = [1.4826 * median(abs.(x .- median(x))) for x in allP]; Em = [median(x) for x in allE]
    Pm, Ps, Em
end

# ------------------------------------------------------------------------------------------------ S/N
"""S/N of series p (1 us): 1-us peak, 10-us running mean, 10-us non-overlapping blocks; baseline = first 20 %; maximum
over all positions and over the bins in rng (the peak window). Returns 6 S/N values and 6 positions (bin of the start)."""
function snr_set(p::Vector{Float64}, rng::UnitRange{Int})
    n = length(p); nb = max(10, floor(Int, 0.2n))
    μ1 = mean(@view p[1:nb]); σ1 = std(@view p[1:nb])
    run = [mean(@view p[i:i+9]) for i in 1:n-9]; μr = mean(@view run[1:nb-9]); σr = std(@view run[1:nb-9])
    nbk = n ÷ 10; p10 = [mean(@view p[10(j-1)+1:10j]) for j in 1:nbk]; nb10 = floor(Int, 0.2nbk)
    μb = mean(@view p10[1:nb10]); σb = std(@view p10[1:nb10])
    r1 = intersect(rng, 1:n); rr = intersect(rng, 1:n-9); rb = intersect(cld(first(rng), 10):fld(last(rng), 10), 1:nbk)
    k1 = argmax(p); kr = argmax(run); kb = argmax(p10)
    q1 = first(r1) - 1 + argmax(@view p[r1]); qr = first(rr) - 1 + argmax(@view run[rr]); qb = first(rb) - 1 + argmax(@view p10[rb])
    vals = [(p[k1] - μ1) / σ1, (run[kr] - μr) / σr, (p10[kb] - μb) / σb, (p[q1] - μ1) / σ1, (run[qr] - μr) / σr, (p10[qb] - μb) / σb]
    pos = [k1, kr, 10(kb - 1) + 1, q1, qr, 10(qb - 1) + 1]
    vals, pos
end

# ------------------------------------------------------------------------------------------------ engines
const WREAD = K.FOLD_WREAD; const PRE = K.FOLD_PRE; const KEEP = K.FOLD_KEEP
function fold_rot_rc(buf::AbstractVector{UInt32}, starts, ig::Float64)
    fplan, iplan = K.plans_for(WREAD)
    [K.process_window_span(view(buf, s0:s0+WREAD-1), ig, PRE+1, KEEP, fplan, iplan; rawcoh=true) for s0 in starts]
end
bin16(RC) = (n = size(RC, 4) ÷ 16; out = zeros(size(RC, 1), size(RC, 2), size(RC, 3), n);
             @inbounds for j = 1:n, k = 1:16; out[:, :, :, j] .+= @view RC[:, :, :, 16(j-1)+k]; end; out)
pow32(RC, h) = (p = dropdims(sum(@view(RC[h, :, :, :]), dims=(1, 2)), dims=(1, 2)); [sum(@view p[9+32(j-1):8+32j]) for j = 1:62])
sumt(RC, r) = dropdims(sum(@view(RC[:, :, :, r]), dims=4), dims=4)
const T0OUT = K.t_start_s + K.edge_crop_ms * 1e-3     # time of output bin 1's start
const PKRNG = (ceil(Int, (K.PK_LO - T0OUT) / 1e-6 + 0.5)):(floor(Int, (K.PK_HI - T0OUT) / 1e-6 + 0.5))

"centred within-channel 4x4 second moments H and cross moments G of the band Stokes S_k with each channel component, bins 1:nb"
function baseline_moments(RC::Array{Float64,4}, xb::Array{Float64,3}, S::NTuple{4,Vector{Float64}}, nb::Int)
    H = zeros(4, 4, NCH, NSB); G = zeros(4, 4, NCH, NSB)
    μS = [mean(@view S[k][1:nb]) for k in 1:4]
    @inbounds for t = 1:nb, s = 1:NSB, c = 1:NCH
        d1 = RC[1, c, s, t] - xb[1, c, s]; d2 = RC[2, c, s, t] - xb[2, c, s]; d3 = RC[3, c, s, t] - xb[3, c, s]; d4 = RC[4, c, s, t] - xb[4, c, s]
        dv = (d1, d2, d3, d4)
        for j = 1:4, i = 1:4; H[i, j, c, s] += dv[i] * dv[j]; end
        for k = 1:4
            sk = S[k][t] - μS[k]
            for i = 1:4; G[k, i, c, s] += sk * dv[i]; end
        end
    end
    H, G
end

function do_catwin(it::Item, buf::Vector{UInt32}, mon)
    cat = parse(Int, it.meta["cat"]); nseq = parse(Int, it.meta["nseq"]); sod = parse(Float64, it.meta["sod"])
    phase = parse(Float64, it.meta["phase"]); detsnr = parse(Float64, it.meta["detsnr"])
    prof, row, ex = K.measure_window(buf, nseq, sod, phase, detsnr, OBSDATE; rawcoh=true)
    ig = 1.0 / K.diffgain_at(sod)
    RC = ex.profiles[9]; nt = size(RC, 4); nb = ex.nb; ilo = ex.ilo; ihi = ex.ihi; ulo = ex.union_lo; uhi = ex.union_hi
    ipk = prof.peak_bin
    gpad = max(3, ihi - ilo + 1); mlo = max(1, ilo - gpad); mhi = min(nt, ihi + gpad)
    slo = max(1, ulo - SPAN_PAD); shi = min(nt, uhi + SPAN_PAD)
    rlo = max(1, ipk - K.RAW_HALFWIN); rhi = min(nt, ipk + K.RAW_HALFWIN)
    RCtot = sumt(RC, 1:nt); RCmask = sumt(RC, mlo:mhi); RCraw = sumt(RC, rlo:rhi)
    xb = ex.RC_base
    S = (ex.profiles[1], ex.profiles[2], ex.profiles[3], ex.profiles[4])
    H, G = baseline_moments(RC, xb, S, nb)                 # function barrier (ex.profiles is not concretely inferred)
    # S/N: uncalibrated per-hand power and Stokes I (table independent), read-time calibrated band I
    PR = vec(sum(@view(RC[1, :, :, :]), dims=(1, 2))); PL = vec(sum(@view(RC[2, :, :, :]), dims=(1, 2)))
    sn = Dict{String,Vector{Float64}}(); sp = Dict{String,Vector{Int}}()
    for (nm, p) in (("R_uncal", PR), ("L_uncal", PL), ("I_uncal", PR .+ PL), ("I_cal_readtime", Vector{Float64}(S[1])))
        v, q = snr_set(p, PKRNG); sn[nm] = v; sp[nm] = q
    end
    # flags (window-internal reference)
    sX = it.off ÷ 4 + floor(Int, K.edge_crop_ms * 1e-3 * K.fs) + (ipk - 1) * K.TS_NAVG
    pb = pulse_bins(mon, (sX,)); Pm, Ps, Em = refs((mon,))
    ok = allfinite(RC) && allfinite(ex.profiles[1])
    fl, maxdev, maxe = flag_one(it.off ÷ 4, it.len ÷ 4, mon, pb, Pm, Ps, Em, ok)
    ok || note_bad(it.id, "non-finite catalogue-window output")
    path = joinpath(ODIR, "items", safe(it.id) * ".h5")
    h5write_atomic(path, h -> begin
        for nt_ in (prof, row), k in keys(nt_)
            v = getfield(nt_, k)
            (v isa Real) && (h["scalars/$(k)"] = Float64(v))
            (v isa AbstractVector{<:Real}) && (h["profiles/$(k)"] = Vector{Float64}(v))
        end
        h["str/nseq"] = string(row.nseq); h["str/utc"] = row.utc === nothing ? "" : Dates.format(row.utc, "yyyy-mm-ddTHH:MM:SS.sss")
        h["scalars/cat"] = cat; h["scalars/ilo"] = ilo; h["scalars/ihi"] = ihi; h["scalars/nb"] = nb; h["scalars/nt"] = nt
        h["scalars/rho_pk"] = prof.Ipk / prof.mI; h["scalars/nseq"] = Float64(nseq); h["scalars/inv_gamma"] = ig
        h["scalars/word0"] = it.off ÷ 4; h["scalars/span_lo"] = slo; h["scalars/span_hi"] = shi
        h["scalars/mask_lo"] = mlo; h["scalars/mask_hi"] = mhi; h["scalars/raw_lo"] = rlo; h["scalars/raw_hi"] = rhi
        h["scalars/n_off"] = count(ex.offmask)
        h["RC_on"] = ex.RC_on; h["RC_on_union"] = ex.RC_on_union; h["RC_off"] = ex.RC_off; h["RC_base"] = ex.RC_base
        h["RC_tot"] = RCtot; h["RC_mask"] = RCmask; h["RC_rawwin"] = RCraw
        h["H_base"] = H; h["G_base"] = G
        h["RC_span"] = Float32.(RC[:, :, :, slo:shi])
        for k in keys(sn); h["snr/$k"] = sn[k]; h["snr/$(k)_pos"] = sp[k]; end
        attributes(h["snr/I_uncal"])["order"] = "1us_peak, run10, block10 (max over window); same over the peak window PK_LO..PK_HI"
        write_mon(h, mon); h["flag/bits"] = fl; h["flag/maxdev_F2"] = maxdev; h["flag/maxratio_F3"] = maxe
        attributes(h)["provenance"] = PROV_READ; attributes(h)["item"] = it.id
    end)
    mark_done(it.id)
end

const FOLDCH = Channel{Any}(Inf)
geom(utc) = (u = DateTime(OBSDATE) + Millisecond(round(Int, utc * 1000)); q = K.parallactic_deg(u); e = K.elevation_deg(u);
             a = K.azimuth_deg(u); (q, e, a, q + K.BWG_ALPHA * e + K.BWG_GAMMA * a))
function do_foldrot(it::Item, buf::Vector{UInt32}, mon)
    starts = parse.(Int, split(it.meta["starts"], ',')); utc = parse(Float64, it.meta["utc_rot"])
    ig = 1.0 / K.diffgain_at(utc)
    r = fold_rot_rc(buf, starts, ig)
    ok = all(allfinite(x[9]) && allfinite(x[1]) for x in r)
    ok || note_bad(it.id, "non-finite fold output")
    res = (id=it.id, rot=parse(Int, it.meta["rot"]), block=it.block, ok=ok, utc=utc, ig=ig, geo=geom(utc),
           rc=[Float32.(bin16(x[9])) for x in r], cal=[hcat(x[1], x[2], x[3], x[4]) for x in r],
           pow=[pow32(r[1][9], 1) pow32(r[1][9], 2) pow32(r[3][9], 1) pow32(r[3][9], 2)],
           mon=mon, w0=it.off ÷ 4, nw=it.len ÷ 4, sX=parse.(Int, split(it.meta["sX"], ',')),
           listed=parse(Int, it.meta["listed"]), gap=parse(Int, it.meta["in_list_gap"]), spiky=parse(Int, it.meta["in_spiky"]))
    put!(FOLDCH, res)
end
function close_block(b, rs)
    sort!(rs, by = r -> r.rot)
    Pm, Ps, Em = refs([r.mon for r in rs])
    flags = zeros(Int, length(rs)); maxdev = zeros(length(rs)); maxe = zeros(length(rs))
    for (i, r) in enumerate(rs)
        flags[i], maxdev[i], maxe[i] = flag_one(r.w0, r.nw, r.mon, pulse_bins(r.mon, (r.sX[1], r.sX[3])), Pm, Ps, Em, r.ok)
    end
    accA = [zeros(size(rs[1].rc[j])) for j = 1:3]; accF = [zeros(size(rs[1].rc[j])) for j = 1:3]
    calA = [zeros(2000, 4) for _ = 1:3]; calF = [zeros(2000, 4) for _ = 1:3]; nA = 0; nF = 0
    for (i, r) in enumerate(rs)                            # rotation order, so the sums are deterministic
        flags[i] == 0 || continue
        nA += 1; for j = 1:3; accA[j] .+= r.rc[j]; calA[j] .+= r.cal[j]; end
        if r.listed == 0 && r.gap == 0
            nF += 1; for j = 1:3; accF[j] .+= r.rc[j]; calF[j] .+= r.cal[j]; end
        end
    end
    path = joinpath(ODIR, "blocks", "block_$(b).h5")
    h5write_atomic(path, h -> begin
        for (j, nm) in enumerate(("MP", "OFF", "LFIP"))
            h["all/$nm/RC16"] = Float32.(accA[j]); h["free/$nm/RC16"] = Float32.(accF[j])
            h["all/$nm/cal"] = calA[j]; h["free/$nm/cal"] = calF[j]
        end
        h["n_all"] = nA; h["n_free"] = nF
        h["rot/rotation"] = [r.rot for r in rs]; h["rot/flag"] = flags; h["rot/listed"] = [r.listed for r in rs]
        h["rot/in_list_gap"] = [r.gap for r in rs]; h["rot/in_spiky"] = [r.spiky for r in rs]
        h["rot/maxdev_F2"] = maxdev; h["rot/maxratio_F3"] = maxe
        h["rot/read_word0"] = [r.w0 for r in rs]
        h["rot/utc"] = [r.utc for r in rs]; h["rot/inv_gamma"] = [r.ig for r in rs]
        h["rot/q_deg"] = [r.geo[1] for r in rs]; h["rot/el_deg"] = [r.geo[2] for r in rs]; h["rot/az_deg"] = [r.geo[3] for r in rs]
        h["rot/chi_bwg_readtime_deg"] = [r.geo[4] for r in rs]
        h["rot/pow32_MP_R_MP_L_LFIP_R_LFIP_L"] = Float32.(cat([r.pow for r in rs]..., dims=3))
        h["ref/P"] = Pm; h["ref/sigP"] = Ps; h["ref/e"] = Em
        h["mon/P"] = Float32.(cat([r.mon.P for r in rs]..., dims=2)); h["mon/e"] = Float32.(cat([r.mon.f0 .+ r.mon.f15 for r in rs]..., dims=2))
        h["mon/n"] = vcat([r.mon.n for r in rs]...); h["mon/bin"] = vcat([r.mon.bin for r in rs]...)
        attributes(h)["provenance"] = PROV_READ; attributes(h)["block"] = b
        attributes(h)["note"] = "RC16 = sum over unflagged rotations of the uncalibrated per-channel coherency (RR, LL, Re RL, Im RL; " *
            "32 ch x 4 subbands) per 16-us bin of each 2-ms window (125 bins), float32; 'free' = rotations without a listed pulse " *
            "(round(Nseq+Phase), any list row) and outside list gaps; cal = band I,Q,U,V at 1 us calibrated with the read-time " *
            "tables (reference only, feed frame)"
    end)
    mark_done("block:$(b)")
    logline("progress.log", @sprintf("%s block %d closed: %d rotations, %d flagged (bit 1 %d, 2 %d, 4 %d, 8 %d), n_all %d, n_free %d\n",
            Dates.format(now(), "HH:MM:SS"), b, length(rs), count(!=(0), flags), count(f -> f & 1 != 0, flags),
            count(f -> f & 2 != 0, flags), count(f -> f & 4 != 0, flags), count(f -> f & 8 != 0, flags), nA, nF))
end

# ------------------------------------------------------------------------------------------------ run
todo_reads = [r for r in reads if any(!isdone, r.items)]
tot_bytes = sum(r.len for r in todo_reads; init=0)
@printf("%s\n%d items (%d reads, %.3f GB) of which %d reads (%.3f GB) to do; %d default threads + %d interactive; queue %d MB\n",
        PROV_READ, length(items), length(reads), sum(r.len for r in reads; init=0) / 1e9, length(todo_reads), tot_bytes / 1e9,
        nthreads(:default), nthreads(:interactive), QMB)
srcpath = SOURCE == :disk ? RAW : SLICES
isfile(srcpath) || error("input not found: $srcpath")
isfile(joinpath(ODIR, "reads.tsv")) || logline("reads.tsv", "t_start_unix\toffset_bytes\tlength_bytes\tread_s\twait_budget_s\tn_items\n")
let w = rand(UInt32, 1_354_825)          # warm-up (compile, plans, operators) on random words
    t = time()
    K.measure_window(w[1:1_280_000], 1, 30000.0, 0.0, 0.0, OBSDATE; rawcoh=true); fold_rot_rc(w, (1, 433336, 874826), 1.0)
    snr_set(rand(8000), PKRNG)
    @printf("warm-up %.1f s\n", time() - t)
end
const SEM = Base.Semaphore(QMB)
isempty(reads) || maximum(cld(r.len, MiB) for r in reads) <= QMB || error("queue budget smaller than one read")
work = Channel{Any}(4 * nthreads(:default) + 8)
bytes_read = Threads.Atomic{Int}(0); read_time = Threads.Atomic{Float64}(0.0); reads_done = Threads.Atomic{Int}(0)
stop_reading = Threads.Atomic{Bool}(false)
T0 = time()
reader = Threads.@spawn :interactive begin
    io = open(srcpath, "r")
    try
        for rd in todo_reads
            stop_reading[] && break
            todo = [it for it in rd.items if !isdone(it)]
            np = cld(rd.len, MiB); tw = time(); for _ = 1:np; Base.acquire(SEM); end; tw = time() - tw
            buf = Vector{UInt32}(undef, rd.len ÷ 4)
            ts = time()
            try
                read_range!(io, buf, srcoff(rd.off, rd.len))
            catch e
                note_bad("read@$(rd.off)", sprint(showerror, e)); stop_reading[] = true
                for _ = 1:np; Base.release(SEM); end
                break
            end
            dtr = time() - ts
            Threads.atomic_add!(bytes_read, rd.len); Threads.atomic_add!(read_time, dtr); Threads.atomic_add!(reads_done, 1)
            logline("reads.tsv", @sprintf("%.6f\t%d\t%d\t%.6f\t%.6f\t%d\n", ts, rd.off, rd.len, dtr, tw, length(todo)))
            put!(work, (rd, todo, buf, np))
        end
    finally
        close(io); close(work)
    end
end
function process_read(rd, todo, buf, np)
    try
        for it in todo
            a = (it.off - rd.off) ÷ 4 + 1; sub = (it.off == rd.off && it.len == rd.len) ? buf : buf[a : a + it.len ÷ 4 - 1]
            mon = monitors(sub, it.off ÷ 4)
            try
                if it.kind == :catwin;      do_catwin(it, sub, mon)
                elseif it.kind == :foldrot; do_foldrot(it, sub, mon)
                end
            catch e
                note_bad(it.id, sprint(showerror, e, catch_backtrace()))
                it.kind == :foldrot && put!(FOLDCH, (id=it.id, block=it.block, ok=false, failed=true))
            end
        end
    finally
        for _ = 1:np; Base.release(SEM); end
    end
end
workers = [Threads.@spawn :default begin
    for (rd, todo, buf, np) in work
        process_read(rd, todo, buf, np)
    end
end for _ = 1:nthreads(:default)]
collector = Threads.@spawn :default begin
    open_blocks = Dict{Int,Vector{Any}}()
    for r in FOLDCH
        if haskey(r, :failed); note_bad(r.id, "fold rotation failed; block $(r.block) left open"); continue; end
        v = get!(() -> Any[], open_blocks, r.block); push!(v, r)
        if length(v) == BLOCK_N[r.block]
            try
                close_block(r.block, v)
            catch e
                note_bad("block:$(r.block)", sprint(showerror, e, catch_backtrace()))
            end
            delete!(open_blocks, r.block)
        end
    end
    isempty(open_blocks) || logline("bad.log", "open blocks at end: $(collect(keys(open_blocks)))\n")
end
monitor_task = Threads.@spawn :interactive begin
    while !istaskdone(reader) || any(!istaskdone, workers)
        sleep(15)
        el = time() - T0; gb = bytes_read[] / 1e9; r = gb / max(el, 1e-9)
        s = @sprintf("%s  %.0f s  reads %d/%d  %.3f/%.3f GB  %.1f MB/s wall (%.1f MB/s disk-only)  ETA %.0f s  done-list %d  nbad %d  maxrss %.2f GB\n",
                     Dates.format(now(), "HH:MM:SS"), el, reads_done[], length(todo_reads), gb, tot_bytes / 1e9, 1e3r,
                     bytes_read[] / 1e6 / max(read_time[], 1e-9), r > 0 ? (tot_bytes / 1e9 - gb) / r : NaN, length(DONE), nbad[], Sys.maxrss() / 2^30)
        print(s); flush(stdout); logline("progress.log", s)
    end
end
wait(reader); foreach(wait, workers)
close(FOLDCH); wait(collector)
wall = time() - T0
s = @sprintf("END: wall %.1f s; %d reads, %.3f GB; disk-only %.1f MB/s (%.1f GB/h); end-to-end %.1f MB/s (%.1f GB/h); nbad %d; stopped %s; maxrss %.2f GB\n",
             wall, reads_done[], bytes_read[] / 1e9, bytes_read[] / 1e6 / max(read_time[], 1e-9), bytes_read[] / 1e9 / max(read_time[], 1e-9) * 3600,
             bytes_read[] / 1e6 / wall, bytes_read[] / 1e9 / wall * 3600, nbad[], stop_reading[], Sys.maxrss() / 2^30)
print(s); logline("progress.log", s)
