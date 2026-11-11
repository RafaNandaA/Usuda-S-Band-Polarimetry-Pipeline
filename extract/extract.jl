# Extract stage: raw 4-bit R/L voltages -> per-pulse uncalibrated store, fold blocks and nebula snapshots.
#
#   julia extract/extract.jl <config.toml> plan             item lists (no raw read): <work_dir>/items.tsv, nebula_items.tsv, fold_blocks.tsv, timing_model.json
#   julia -t 5,1 extract/extract.jl <config.toml> windows   catalogue windows and fold rotations -> <work_dir>/items/, blocks/
#   julia -t 4 extract/extract.jl <config.toml> nebula      nebula snapshots -> [paths] nebula_file, gain_step_file
#
# Options (windows, nebula):
#   --items FILE        item list to read (default <work_dir>/items.tsv for windows)
#   --outdir DIR        output directory for windows (default <work_dir>, read by calibrate/apply.jl)
#   --queue-mb N        RAM budget for read buffers in MB (default 300)
#   --max-items N       read only the first N items (in file order)
#   --source local --slices FILE --slice-map TSV
#                       read the item bytes from a local file holding copies of selected ranges of the raw file
#                       (columns of the map: word0, nwords, local_byte0), for runs without the raw file
# A stopped run is resumed by running the same command again (done items are listed in done.list).
# The raw file is only read, never written.

using Printf, Statistics, HDF5, FFTW, Base.Threads, LinearAlgebra, Dates
FFTW.set_num_threads(1)

const USAGE = "usage: julia [-t 5,1] extract/extract.jl <config.toml> plan|windows|nebula [options]"
length(ARGS) >= 2 || error(USAGE)
const CONFIG = abspath(ARGS[1]); const MODE = ARGS[2]
MODE in ("plan", "windows", "nebula") || error(USAGE)
ENV["USUDAPOL_CONFIG"] = CONFIG
include(joinpath(@__DIR__, "lib", "config.jl"))
const CFG = load_config(CONFIG)
getarg(n, d) = (i = findfirst(==(n), ARGS); i === nothing ? d : ARGS[i+1])

module K
include(joinpath(@__DIR__, "lib", "fold_kernel.jl"))
end

const WORK = CFG["paths"]["work_dir"]
const ITEMS = getarg("--items", joinpath(WORK, "items.tsv"))
const ODIR = getarg("--outdir", WORK)
const SOURCE = Symbol(getarg("--source", "disk"))
const RAW = CFG["paths"]["raw_file"]
const SLICES = getarg("--slices", ""); const SLICEMAP = getarg("--slice-map", "")
SOURCE in (:disk, :local) || error("--source must be disk or local")
SOURCE == :local && (isfile(SLICES) && isfile(SLICEMAP) || error("--source local needs --slices FILE and --slice-map TSV"))
const QMB = parse(Int, getarg("--queue-mb", "300"))
const MAXITEMS = parse(Int, getarg("--max-items", "0"))

include(joinpath(@__DIR__, "lib", "rawio.jl"))
if MODE == "plan"
    run(`$(get(ENV, "PYTHON", "python3")) -I $(joinpath(@__DIR__, "lib", "plan_items.py")) $CONFIG`)
    include(joinpath(@__DIR__, "lib", "nebula.jl"))
    plan_nebula()
elseif MODE == "windows"
    include(joinpath(@__DIR__, "lib", "reader.jl"))
else
    include(joinpath(@__DIR__, "lib", "nebula.jl"))
    run_nebula()
end
