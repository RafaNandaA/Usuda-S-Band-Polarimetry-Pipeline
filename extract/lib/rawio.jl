# Raw-file access shared by the window reader and the nebula snapshots: item metadata, byte ranges, local slice files and
# the 1-ms monitors of the digitiser streams.

const MiB = 1 << 20
const CHUNK = 16 * MiB
const WPB = 64_000                       # words per 1-ms monitor bin

function parse_meta(s)
    d = Dict{String,String}()
    for kv in split(s, ';'); i = findfirst('=', kv); i === nothing && continue; d[kv[1:i-1]] = kv[i+1:end]; end
    d
end
safe(id) = replace(id, ':' => '_')

# ------------------------------------------------------------------------------------------------ source mapping
# --source local reads the same bytes from a local file holding copies of selected ranges of the raw file
# (--slices FILE, --slice-map TSV with columns word0, nwords, local_byte0), for runs without the raw file.
const LMAP = Tuple{Int,Int,Int}[]
if SOURCE == :local
    for ln in readlines(SLICEMAP)[2:end]
        t = split(ln, '\t'); push!(LMAP, (parse(Int, t[1]), parse(Int, t[2]), parse(Int, t[3])))
    end
end
function srcoff(off::Int, len::Int)
    SOURCE == :disk && return off
    w = off ÷ 4; n = len ÷ 4
    for (w0, nw, b0) in LMAP
        (w0 <= w && w + n <= w0 + nw) && return b0 + 4 * (w - w0)
    end
    error("range $off+$len not in the local slice file")
end
function read_range!(io, buf::Vector{UInt32}, off::Int)
    nb = sizeof(buf); seek(io, off); pos = 0
    GC.@preserve buf begin
        p = Ptr{UInt8}(pointer(buf))
        while pos < nb
            m = min(CHUNK, nb - pos)
            unsafe_read(io, p + pos, m)            # throws EOFError on a short read
            pos += m
        end
    end
    return pos
end

# ------------------------------------------------------------------------------------------------ 1-ms monitors
"per 1-ms bin of the file grid and per stream (R0-3, L0-3): mean power of the levels, fraction at level 0 and at 15"
function monitors(buf::AbstractVector{UInt32}, w0::Int)
    b0 = fld(w0, WPB); b1 = fld(w0 + length(buf) - 1, WPB); nb = b1 - b0 + 1
    S = zeros(8, nb); N0 = zeros(Int, 8, nb); N15 = zeros(Int, 8, nb); n = zeros(Int, nb)
    @inbounds for b = 1:nb
        a = max(1, (b0 + b - 1) * WPB - w0 + 1); e = min(length(buf), (b0 + b) * WPB - w0)
        n[b] = e - a + 1
        s1 = zeros(8); c0 = zeros(Int, 8); c15 = zeros(Int, 8)
        for i = a:e
            w = buf[i]
            for ch = 0:7
                v = (w >> (4ch)) & 0x0000000F
                x = Float64(v) - 7.5; s1[ch+1] += x * x
                c0[ch+1] += (v == 0); c15[ch+1] += (v == 15)
            end
        end
        S[:, b] .= s1 ./ n[b]; N0[:, b] .= c0; N15[:, b] .= c15
    end
    return (bin = collect(b0:b1), n = n, P = S, f0 = N0 ./ n', f15 = N15 ./ n')
end
function write_mon(h, m, g="mon")
    h["$g/bin"] = m.bin; h["$g/n"] = m.n; h["$g/P"] = m.P; h["$g/f0"] = m.f0; h["$g/f15"] = m.f15
end
