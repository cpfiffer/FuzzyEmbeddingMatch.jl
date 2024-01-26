# README for FuzzyEmbeddingMatch

## Overview
The `FuzzyEmbeddingMatch` module is designed to facilitate fuzzy string matching by leveraging embeddings. It primarily consists of structures and functions to embed strings, calculate similarities between these embeddings, and find the best or all matches within a set of candidates. Key components include `EmbeddedString`, `MatchCandidate`, `bestmatch`, and `allmatches`.

This module uses memoization for embedding strings to reduce API calls.

## Installation

You can install this package with

```julia
import Pkg
Pkg.add("FuzzyEmbeddingMatch")
```

or, from the REPL:

```julia
] add FuzzyEmbeddingMatch
```

## Usage

To begin, make sure that your environment variable `OPENAI_API_KEY` is set. If you do not have the environment variable set at the system level, you can add it with

```julia
ENV["OPENAI_API_KEY"] = "........" # Replace this with your key
```

### Structures
- `EmbeddedString`: Represents a string with its associated embedding.
- `MatchCandidate`: A candidate for matching, containing two strings, their embeddings, and a similarity score.

### Key Functions
- `embed`: Embeds a string using `aiembed` from `PromptingTools.jl`.
- `corpus`: Generates a corpus of embedded strings.
- `getembeddings`: Returns embeddings for a vector of strings.
- `cosinesimilarity`: Calculates cosine similarity between two embeddings.

### Matching Functions
- `allmatches`: Finds all matches for a given string in a list of candidates.
- `bestmatch`: Finds the best match for a given string in a list of candidates.

## Examples

### Using `allmatches`
```julia
# Example strings and candidates
thing = "Example string"
candidates = ["Sample text", "Example string", "Another example"]

# Finding all matches
matches = allmatches(thing, candidates)

# Output the matches
for match in matches
    println(match)
end
```

### Using `bestmatch`
```julia
# Example string and candidates
thing = "Example string"
candidates = ["Sample text", "Example string", "Another example"]

# Finding the best match
best_match = bestmatch(thing, candidates)

# Output the best match
println("Best match: ", best_match)
```

## Notes
- 
- The `cosinesimilarity` function is a key component in computing the match score.
- ProgressMeter is utilized in `getembeddings` for visual progress indication.

---

Ensure to have the required Julia packages installed and import the `FuzzyEmbeddingMatch` module to begin using these functionalities in your projects.
