using ArgParse
using Distances
using Clustering
using Statistics

function args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        # "--in"
        #     help = "an option with an argument"
        # "--opt2", "-o"
        #     help = "another option with an argument"
        #     arg_type = Int
        #     default = 0
        # "--flag1"
        #     help = "an option without argument, i.e. a flag"
        #     action = :store_true
        "infile"
            help = "the input data"
            required = true
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
            d = dist(other_centroid, data[point, :])
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

data = init(args())
choice, d = p_dispersion(data, 5)
println("minimum distance in the chosen points: $d")
println("chosen points:")
println(choice)
