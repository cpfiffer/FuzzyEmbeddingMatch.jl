module FuzzyEmbeddingMatch

# Exports
export EmbeddedString, MatchCandidate, bestmatch, allmatches, corpus

using PromptingTools
using LinearAlgebra
using Memoize
using ProgressMeter

"""
    EmbeddedString{T,E}

A string `content` with an embedding `embedding`.
"""
struct EmbeddedString{T,E}
    content::T
    embedding::E
end
function EmbeddedString(content::AbstractString)
    EmbeddedString(content, embed(content))
end

"""
Overrides the `show` function for `EmbeddedString` to only show the
`content` of the string.
"""
function Base.show(io::IO, x::EmbeddedString)
    print(io, "EmbeddedString(\"", x.content, "\")")
end


"""
    embed(x::AbstractString)

Return the embedding of `x` using `aiembed` from PromptingTools.jl.
Memoized to reduce the number of API calls to `aiembed`.
"""
@memoize embed(x::AbstractString) = aiembed(x; verbose=false).content

"""
    MatchCandidate{T,E}

A candidate match between two strings `content_a` and `content_b` with
embeddings `embedding_a` and `embedding_b` and a score `score`.
"""
struct MatchCandidate{T,E}
    content_a::T
    content_b::T
    embedding_a::E
    embedding_b::E
    score::Float64
end
function MatchCandidate(
    a::EmbeddedString,
    b::EmbeddedString,
)
    MatchCandidate(
        a.content,
        b.content,
        a.embedding,
        b.embedding,
        cosinesimilarity(a, b)
    )
end

# Show method override for `MatchCandidate`.
function Base.show(io::IO, x::MatchCandidate)
    print(io, "MatchCandidate(\"", x.content_a, "\", \"", x.content_b, "\", ", x.score, ")")
end

"""
    corpus(things::AbstractVector{<:AbstractString})

Return the corpus of all the strings in `things`. Used to cache the
embeddings of all the strings in `things` to reduce the number of 
API calls to `aiembed`.
"""
function corpus(things::AbstractVector{<:AbstractString})
    # Get the embeddings of all the strings in `things`.
    embeds = getembeddings(things)

    # Return the corpus of all the embeddings.
    return embeds
end

"""
    getembeddings(things::AbstractVector{<:AbstractString})

Return the embeddings of each string in `things`.
"""
function getembeddings(things::AbstractVector{<:AbstractString})
    embeds = @showprogress map(unique(things)) do x
        EmbeddedString(x)
    end

    # Return the embeddings in their original order.
    return map(x -> embeds[findfirst(y -> y.content == x, embeds)], things)
end

"""
    bestmatch(
        thing::AbstractString,
        candidates::AbstractVector{<:AbstractString};
        threshold=0.5
    )

Return the best match for `thing` in `candidates` by comparing the embedding
of `thing` to the embeddings of `candidates`. The embeddings are computed
using `aiembed` from PromptingTools.jl.
"""
function bestmatch(
    thing::AbstractString,
    candidates::AbstractVector{<:AbstractString};
    verbose=false
)
    # Get all matches
    matches = allmatches(thing, candidates)

    # Sort the matches by score.
    sort!(matches, by=x -> x.score, rev=true)

    # Return the best match.
    return matches[1]
end

"""
    allmatches(
        thing::AbstractString,
        candidates::AbstractVector{<:AbstractString};
        threshold=0.5
    )

Return all the matches for `thing` in `candidates` by comparing the embedding
of `thing` to the embeddings of `candidates`. The embeddings are computed
using `aiembed` from PromptingTools.jl.
"""
function allmatches(
    thing::AbstractString,
    candidates::AbstractVector{<:AbstractString};
)

    # Get the embeddings of `thing` and `candidates`.
    thing_embed = EmbeddedString(thing)
    candidate_embeds = getembeddings(candidates)

    # Compare all the embeds
    return map(x -> MatchCandidate(thing_embed, x), candidate_embeds)
end

"""
    cosinesimilarity(a::AbstractVector, b::AbstractVector)

Compare the embeddings of `a` and `b` by computing the cosine similarity.
"""
cosinesimilarity(a::EmbeddedString, b::EmbeddedString) = cosinesimilarity(a.embedding, b.embedding)
function cosinesimilarity(a::AbstractVector, b::AbstractVector)
    # Compute the dot product of `a` and `b`.
    dot_product = dot(a, b)

    # Compute the norm of `a`.
    norm_a = norm(a)

    # Compute the norm of `b`.
    norm_b = norm(b)

    # Compute the cosine similarity.
    dot_product / (norm_a * norm_b)
end

end # module FuzzyEmbeddingMatch
