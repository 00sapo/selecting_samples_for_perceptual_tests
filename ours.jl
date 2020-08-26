using ArgParse
using Distances
using Clustering
using Statistics
using Plots
using NPZ
include("contardo/PDispersion.jl")

function args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "infile"
            help = "the input data"
            required = false
    end

    parsed_args = parse_args(ARGS, s)
    path = parsed_args["infile"]
end

function init(path)
    local tsp
    open(path) do f
        SPLIT = false
        for line in eachline(f)
            splits = split(line)
            if splits[1] == "DIMENSION"
                tsp = Array{Float32, 2}(undef, parse(Int, splits[3]), 2)
            elseif splits[1] == "EOF"
                break
            end
            if SPLIT
                tsp[parse(Int, splits[1]), :] = [parse(Float32, splits[2]), parse(Float32, splits[3])]
            end
            # this happens only once, when split is still false
            if splits[1] == "NODE_COORD_SECTION"
                SPLIT = true
            end
        end
    end

    return tsp
end

function p_dispersion(data, p::Integer)
    dist = Euclidean()
    D = pairwise(dist, data, data, dims=1)
    result = hclust(D, linkage=:ward)
    clusters = cutree(result, k=p)
    choice = Array{Integer, 1}(undef, p)
    for i in 1:p
        this_idx = findall(x -> x == i, clusters)
        max_dist = -Inf
        local chosen
        for point in this_idx
            other_centroid = mean([data[1:point-1, :]; data[point+1:end, :]], dims=1)
            d = mean(abs.(other_centroid - data[point, :]'))
            # d = dist(other_centroid, data[point, :]')
            if d > max_dist
                max_dist = d
                chosen = point
            end
        end
        choice[i] = chosen
    end

    D = pairwise(dist, data[choice, :], data[choice, :], dims=1)

    return choice, minimum(D[D .!= 0])
end

data = npzread("./samples_PCA_2.npy")
choice, d = p_dispersion(data, 4)
PDispersion.data.D = data'
PDispersion.data.nnodes = size(data)[1]
lb, ub, opt, _, _ = PDispersion.pdispersion_decremental_clustering(4)

pgfplotsx()
scatter(data[:, 1], data[:, 2], color="white")
scatter!(data[choice, 1], data[choice, 2], color="red")
scatter!(data[opt, 1], data[opt, 2], color="blue")
savefig("plot.svg")
gui()

println("Found distance (ours, lb, ub): $d, $lb, $ub")
println("\nFound points ours, contardo")
println(choice)
println(opt)
println()
