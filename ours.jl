using Distances
using Clustering
using Statistics
using Plots
using NPZ
using Printf
using Suppressor
include("contardo/PDispersion.jl")

# Gurobi license file
ENV["GRB_LICENSE_FILE"] = joinpath(homedir(), "Templates/gurobi.lic")

function init(path)
    local tsp
    open(path) do f
        SPLIT = false
        for line in eachline(f)
            splits = split(line)
            if splits[1] == "DIMENSION"
                tsp = Array{Float32,2}(undef, parse(Int, splits[3]), 2)
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

"""
Finds p clusters and for each chose the point which maximize the minimum
distance from all the other points in the dataset
"""
function p_dispersion_minmax(data, p::Integer)
    dist = Euclidean()
    D = pairwise(dist, data, data, dims=1)
    result = hclust(D, linkage=:ward)
    clusters = cutree(result, k=p)
    choice = Array{Integer,1}(undef, p)
    for i in 1:p
        # compute the max-min distance against single points
        this_idx = findall(x -> x == i, clusters)
        max_min_dist = -Inf
        local chosen
        for point in this_idx
            others = [data[1:point - 1, :]; data[point + 1:end, :]]
            min_dist = Inf
            for other in eachrow(others)
                d = sum(abs.(other - data[point, :]))
                if d < min_dist
                    min_dist = d
                end
            end
            if min_dist > max_min_dist
                max_min_dist = min_dist
                chosen = point
            end
        end
        choice[i] = chosen
    end

    return choice
end

"""
Finds p clusters and for each chose the point which maximize the
distance from the centroid of the other points in the dataset
"""
function p_dispersion_centroid(data, p::Integer)
    dist = Euclidean()
    D = pairwise(dist, data, data, dims=1)
    result = hclust(D, linkage=:ward)
    clusters = cutree(result, k=p)
    choice = Array{Integer,1}(undef, p)
    for i in 1:p
        this_idx = findall(x -> x == i, clusters)
        max_dist = -Inf
        local chosen
        for point in this_idx
            other_centroid = mean([data[1:point - 1, :]; data[point + 1:end, :]], dims=1)
            d = mean(abs.(other_centroid - data[point, :]'))
            # d = dist(other_centroid, data[point, :]')
            if d > max_dist
                max_dist = d
                chosen = point
            end
        end
        choice[i] = chosen
    end

    return choice
end

"""
Finds p clusters and for each chose the point which maximize the minimum
distance from the other clusters' points
"""
function p_dispersion_other_minmax(data, p::Integer)
    dist = Euclidean()
    D = pairwise(dist, data, data, dims=1)
    result = hclust(D, linkage=:ward)
    clusters = cutree(result, k=p)
    choice = Array{Integer,1}(undef, p)
    for i in 1:p
        # compute the max-min distance against single points
        this_idx = findall(x -> x == i, clusters)
        not_this_idx = findall(x -> x != i, clusters)
        max_min_dist = -Inf
        local chosen
        for point in this_idx
            min_dist = Inf
            for j in not_this_idx
                d = sum(abs.(data[j, :] - data[point, :]))
                if d < min_dist
                    min_dist = d
                end
            end
            if min_dist > max_min_dist
                max_min_dist = min_dist
                chosen = point
            end
        end
        choice[i] = chosen
    end

    return choice
end

"""
Finds p clusters and for each chose the point which maximize the minimum
distance from the centroids of the other clusters
"""
function p_dispersion_other_centroid_minmax(data, p::Integer)
    dist = Euclidean()
    D = pairwise(dist, data, data, dims=1)
    result = hclust(D, linkage=:ward)
    clusters = cutree(result, k=p)
    choice = Array{Integer,1}(undef, p)
    # computing centroids of each cluster
    centroids = zeros(p, size(data)[2])
    for i in 1:p
        idx = findall(x -> x == i, clusters) 
        centroids[i, :] = mean(data[idx, :], dims=1)
    end
    for i in 1:p
        # compute the max-min distance against centroids
        this_idx = findall(x -> x == i, clusters)
        max_min_dist = -Inf
        local chosen
        for point in this_idx
            min_dist = Inf
            for j in 1:p
                if i == j
                    continue
                else
                    d = sum(abs.(centroids[j, :] - data[point, :]))
                    if d < min_dist
                        min_dist = d
                    end
                end
            end
            if min_dist > max_min_dist
                max_min_dist = min_dist
                chosen = point
            end
        end
        choice[i] = chosen
    end

    return choice 
end

"""
Copmuting the minimum distance without using pairwise which is subject to
errors due to the limited number precisions and sometimes returnes values != 0
along the diagonal
"""
function mindist(data, choice)
    C = convert(Array{Float64}, data[choice, :])
    n = size(C)[1]
    min = Inf
    for i in 1:n, j in 1:n
        if i == j
            continue
        end
        d = euclidean(C[i, :], C[j, :])
        if d < min
            min = d
        end
    end
    return min
end

function plot(data, choices)
    # pgfplotsx()
    gr()
    choiceA, choiceB, choiceC, choiceD, opt = choices
    scatter(data[:, 1], data[:, 2], color="white", label="Data", markershape=:circle, legend=:topleft)
    scatter!(data[choiceA, 1], data[choiceA, 2], color="red", label="Method A", markershape=:circle, markeralpha=1.0)
    scatter!(data[choiceB, 1], data[choiceB, 2], color="orange", label="Method B", markershape=:xcross, markeralpha=1.0)
    scatter!(data[choiceC, 1], data[choiceC, 2], color="green", label="Method C", markershape=:utriangle, markeralpha=1.0)
    # scatter!(data[choiceD, 1], data[choiceD, 2], color="yellow", label="Method D", markershape=:xcross, markeralpha=1.0)
    scatter!(data[opt, 1], data[opt, 2], color="blue", label="Contardo", marker=:cross, markeralpha=1.0)

end

function main()
    Ps = [4, 5, 10, 20]
    datadir = "./data"
    paths = readdir(datadir)
    # data = npzread("./samples_PCA_2.npy")
    println("Minimum distances (and time) in the chosen points: ")
    println("dataset\t\t| P  | choiceA\t\t | choiceB\t\t | choiceC\t\t | choiceD\t\t | contardo\t\t |")
    for p in Ps, file in paths
        path = joinpath(datadir, file)
        if file[end-3:end] == ".npy"
            data = npzread(path)
        else
            continue
            data = init(path)
        end
        tA = @elapsed choiceA = p_dispersion_centroid(data, p)
        tB = @elapsed choiceB = p_dispersion_other_centroid_minmax(data, p)
        tC = @elapsed choiceC = p_dispersion_other_minmax(data, p)
        tD = @elapsed choiceD = p_dispersion_minmax(data, p)
        PDispersion.data.D = data'
        PDispersion.data.nnodes = size(data)[1]
        PDispersion.set_params(:euclidean, 300)
        t_cont = @elapsed lb, ub, opt, groups, avgSize = @suppress PDispersion.pdispersion_decremental_clustering(p)

        dA = mindist(data, choiceA)
        dB = mindist(data, choiceB)
        dC = mindist(data, choiceC)
        dD = mindist(data, choiceD)
        d_cont = mindist(data, opt)
        @printf("%s\t| %2d | %.1e(%.1e)\t | %.1e(%.1e)\t | %.1e(%.1e)\t | %.1e(%.1e)\t | %.1e(%.1e)\t |\n", basename(file), p, dA, tA, dB, tB, dC, tC, dD, tD, d_cont, t_cont)
        if file[end-3:end] == ".npy"
            plot(data, (choiceA, choiceB, choiceC, choiceD, opt))
            gui()
            savefig("plot$p.svg")
        end
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
