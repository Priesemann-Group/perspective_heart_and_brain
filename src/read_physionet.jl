# --- Structures to hold header information ---

# Information about a single signal channel.
struct WFDBSignalInfo
    filename::String    # e.g. "426_C_VF_454_5s_frag.dat"
    format::String      # e.g. "16" or "212"
    gain::Float64       # scaling factor
    baseline::Float64   # baseline offset
end

# Information from the header file.
struct WFDBRecordHeader
    record_name::String
    nsig::Int         # number of signals (channels)
    fs::Float64       # sampling frequency (Hz)
    nsamp::Union{Int, Nothing}  # total number of samples (if given)
    signals::Vector{WFDBSignalInfo}
end

# --- Functions for reading the header file ---

"""
    read_header(header_file::String) -> WFDBRecordHeader

Reads a WFDB header file (with extension ".hea") and returns a WFDBRecordHeader.
Assumes that the first line is formatted as:
    
    record_name nsig fs [nsamp]

and that each subsequent line describes one signal with at least these fields:
    
    filename format gain baseline

Any extra tokens are ignored.
"""
function read_header(header_file::String)::WFDBRecordHeader
    if !isfile(header_file)
        error("Header file not found: $header_file")
    end
    lines = readlines(header_file)
    if isempty(lines)
        error("Empty header file: $header_file")
    end

    # Parse first line.
    first_line_tokens = split(lines[1])
    record_name = first_line_tokens[1]
    nsig = parse(Int, first_line_tokens[2])
    fs = parse(Float64, first_line_tokens[3])
    nsamp = nothing
    if length(first_line_tokens) >= 4
        try
            nsamp = parse(Int, first_line_tokens[4])
        catch
            nsamp = nothing
        end
    end

    # Parse each subsequent line for signal info.
    signals = WFDBSignalInfo[]
    for i in 2:length(lines)
        tokens = split(lines[i])
        if length(tokens) < 4
            continue  # skip lines that don’t have enough fields
        end
        sig_filename = tokens[1]
        sig_format = tokens[2]   # e.g. "212" or "16"
        sig_gain = parse(Float64, tokens[3])
        sig_baseline = parse(Float64, tokens[4])
        push!(signals, WFDBSignalInfo(sig_filename, sig_format, sig_gain, sig_baseline))
    end

    return WFDBRecordHeader(record_name, nsig, fs, nsamp, signals)
end

# --- Functions to read the .dat file in different formats ---

"""
    read_16_format(dat_file::String) -> Vector{Int16}

Reads a binary file in the “16” format: a stream of 16‐bit two’s complement integers.
"""
function read_16_format(dat_file::String)::Vector{Int16}
    if !isfile(dat_file)
        error("Data file not found: $dat_file")
    end
    data = read(dat_file)
    # reinterpret the raw bytes as Int16 values (assumes little‐endian; adjust if needed)
    samples = collect(reinterpret(Int16, data))
    return samples
end

"""
    read_212_format(dat_file::String) -> Vector{Int16}

Reads a binary file in the “212” format.
In this format, every 3 bytes encode two 12‐bit samples.
"""
function read_212_format(dat_file::String)::Vector{Int16}
    if !isfile(dat_file)
        error("Data file not found: $dat_file")
    end
    bytes = read(dat_file)
    n = length(bytes)
    n_triplets = div(n, 3)  # number of complete triplets
    n_samples = n_triplets * 2
    samples = Vector{Int16}(undef, n_samples)
    j = 1
    for i in 1:3:(n_triplets * 3)
        b1 = bytes[i]
        b2 = bytes[i+1]
        b3 = bytes[i+2]
        # Decode two 12-bit samples:
        # Sample 1: low byte b1 and low nibble of b2.
        s1 = Int16(b1) + (Int16(b2 & 0x0F) << 8)
        # Sample 2: low byte b3 and high nibble of b2.
        s2 = Int16(b3) + (Int16((b2 & 0xF0) >> 4) << 8)
        # Adjust for sign (12-bit two’s complement).
        if s1 >= 2048
            s1 -= 4096
        end
        if s2 >= 2048
            s2 -= 4096
        end
        samples[j] = s1
        samples[j+1] = s2
        j += 2
    end
    return samples
end
# --- Function to read an entire WFDB record ---

"""
    read_wfdb_record(record_path::String) -> (header, signal)

Given the base record path (without extension) for a WFDB record—for example,
"ecg-fragment-high-risk-label/1.0.0/1_Dangerous_VFL_VF/frag/426_C_VF_454_5s_frag"—
this function reads the associated “.hea” and “.dat” files.

For a single‐channel record, it returns the header and a vector of the scaled signal
(as Float64 values in physical units, computed as `(raw – baseline)/gain`).
For multichannel records the data are assumed to be interleaved (this example
only implements scaling for each channel and returns a vector per channel).
"""
function read_wfdb_record(record_path::String)
    # The header file is record_path + ".hea"
    header_file = record_path * ".hea"
    header = read_header(header_file)
    # The data file is record_path + ".dat"
    dat_file = record_path * ".dat"

    if header.nsig == 1
        # For one channel, read using the format indicated.
        sig_format = header.signals[1].format
        raw_data = if sig_format == "16"
            read_16_format(dat_file)
        elseif sig_format == "212"
            read_212_format(dat_file)
        else
            error("Unsupported data format: $sig_format")
        end
        # Apply scaling: physical value = (raw – baseline) / gain.
        gain = header.signals[1].gain
        baseline = header.signals[1].baseline
        phys = [ (x - baseline) / gain for x in raw_data ]
        return header, phys
    else
        # For multiple signals, the data are interleaved.
        # (Here we assume all signals use the same format.)
        sig_format = header.signals[1].format
        raw_data = if sig_format == "16"
            read_16_format(dat_file)
        elseif sig_format == "212"
            read_212_format(dat_file)
        else
            error("Unsupported data format: $sig_format")
        end

        n_samples = div(length(raw_data), header.nsig)
        signals = [ Vector{Float64}(undef, n_samples) for _ in 1:header.nsig ]
        for i in 1:n_samples
            for j in 1:header.nsig
                raw_val = raw_data[(i-1)*header.nsig + j]
                gain = header.signals[j].gain
                baseline = header.signals[j].baseline
                signals[j][i] = (raw_val - baseline) / gain
            end
        end
        return header, signals
    end
end