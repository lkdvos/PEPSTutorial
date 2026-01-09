### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 7d26a11c-f98c-473a-9d11-9c564deb1272
begin
	using TensorKit
end

# ╔═╡ 9b2b4b2e-eb14-11f0-b4d4-79004a9b7521
md"""
# CTMRG for infinite PEPS with TensorKit.jl

In this tutorial, we aim to write the code to contract infinite networks through the CTMRG algorithm.
This notebook contains an example setup for implementing this algorithm, where some parts are left as exercise to the reader, while others are filled in to speed up the process.
At the end of this tutorial, we want to be able to:

- contract the norm-network of an infinite PEPS $\langle\psi|\psi\rangle$
- compute expectation values of (local) observables for a given PEPS

To accomplish this, we will leverage TensorKit.jl to provide the building blocks for creating the algorithm, and we will assume some familiarity with the methods defined in that package.
For more information on TensorKit and the primitive functions that we will need here, we refer back to the tutorial on `TensorMap`s.

### Outline

We will set up the algorithm as follows, following the typical steps of CTMRG:

1. Introduction and conventions
2. Initialization of the tensors
3. Unidirectional CTMRG move
4. The full CTMRG algorithm
5. Expectation values
6. Conclusions and extensions
"""

# ╔═╡ 44833ecc-202c-40bd-b761-dbd25c0f6f7b
md"""
## Introduction and conventions

### What does CTMRG compute?

Given an infinite 2D tensor network (e.g. the PEPS norm network), the exact contraction is intractable.
CTMRG approximates the infinite environment around a local region by a finite set of tensors:

- Four **corner tensors**: `C₁, C₂, C₃, C₄`
- Four **edge tensors**: `T₁, T₂, T₃, T₄`

![CTMRG contraction](https://github.com/lkdvos/PEPSTutorial/blob/main/notebooks/assets/ctmrg_contraction.jpeg?raw=true)

These tensors represent an effective boundary that *summarizes* the infinite remainder of the lattice.
Once converged, this environment can be contracted with and without local insertions of operators to compute:
- norms and expectation values,
- energy densities for local Hamiltonians,
- horizontal and vertical correlation functions via effective transfer operators.

In practice, CTMRG is an iterative fixed-point method:
- Each move **grows** the boundary by absorbing one layer of the bulk network,
- Then **compresses** it back to bond dimension `χ` using an isometry derived from an effective reduced density matrix (or equivalent construction),
- Repeats until the boundary tensors stop changing (up to normalization / gauge).
"""

# ╔═╡ df08983e-dab0-4a00-bf0c-4eb91f42f1fa
md"""
### Tensors and conventions

Converting this into code, we start by setting up some conventions for the tensors.
"""

# ╔═╡ 20f29190-2e0d-42e3-af82-11761478d29d
"""
    const PEPSTensor{T, S}

Default type for PEPS tensors of element type `T` and space type `S`, with a single physical index, and 4 virtual indices, conventionally ordered as: ``T : P ← N ⊗ E ⊗ S ⊗ W``.
Here, ``P`` denotes the physical space and ``N``, ``E``, ``S`` and ``W`` denote the north, east, south and west virtual spaces, respectively.

```
           N
          ╱
         ╱
   W---- ----E
       ╱|
      ╱ |
     S  P
```
"""
const PEPSTensor{T, S} = AbstractTensorMap{T, S, 1, 4}

# ╔═╡ ab8cc3a0-0fd4-4f11-be1d-40b2a44f53d2
md"""
To make life a little easier in the remainder of this notebook, we will simplify the code by first mapping our double-layered network to a single-layer one, merging the PEPS tensors from the bra and ket together.
"""

# ╔═╡ d2c17a80-1c4a-49be-b615-ed6eb53b0779
const DoublePEPSTensor{T, S} = AbstractTensorMap{T, S, 2, 2}

# ╔═╡ a3a87f5e-da1a-45f2-b45b-8df484a30de8
function merge_braket(bra::PEPSTensor, ket::PEPSTensor)
	V_north = space(bra, 2) ⊗ space(ket, 2)'
	fuse_north = isomorphism(fuse(V_north) ← V_north)
	V_east = space(bra, 3) ⊗ space(ket, 3)'
	fuse_east = isomorphism(fuse(V_east) ← V_east)
	@tensor braket[N E; S W] := conj(bra[p; n' e' s' w']) * ket[p; n e s w] *
		fuse_north[N; n n'] * fuse_east[E; e e'] * conj(fuse_north[S; s s']) * conj(fuse_east[W; w w'])
	return braket
end

# ╔═╡ 9da0bb7d-f918-40a1-871f-a546d7b5f641
md"""
As a result, we can define the CTMRG environment tensors immediately for this double-layer object, leading to a 3-index tensor for the edges, and a 2-index tensor for the corners.
"""

# ╔═╡ c0862aa4-715b-412a-97ec-5bdf00a5fbdf
const EdgeTensor{T, S} = AbstractTensorMap{T, S, 2, 1}

# ╔═╡ f2a65413-604c-45f4-99cc-e4f9abc914be
const CornerTensor{T, S} = AbstractTensorMap{T, S, 1, 1}

# ╔═╡ 4cb24d7c-e175-4cab-90f1-dc91a1d84ec1
md"""
Using this, we can set up the datastructures that we will use as follows:
"""

# ╔═╡ 5d501f80-2e05-48ed-b62e-52ed723a2e90
struct InfinitePEPS{A <: PEPSTensor}
    tensor::A
end

# ╔═╡ 076f533f-1e61-4172-b118-d0815e8c66cd
struct InfiniteDoublePEPS{A <: DoublePEPSTensor}
    tensor::A
end

# ╔═╡ 82fa0408-7690-41db-8ced-823aeb662974
function merge_braket(peps::InfinitePEPS)
    return InfiniteDoublePEPS(merge_braket(peps.tensor, peps.tensor))
end

# ╔═╡ b2b002e3-5c08-4cc6-89c3-07870bf666d7
struct CTMRGEnvironment{E <: EdgeTensor, C <: CornerTensor}
	edge_north::E
	edge_east::E
	edge_south::E
    edge_west::E
	
	corner_northwest::C
	corner_northeast::C
	corner_southeast::C
	corner_southwest::C
end

# ╔═╡ 7a94202f-3487-47c7-923a-61a178257374
md"""
Throughout this tutorial, we will use $d$ to refer to the dimension of the physical space, $D$ for dimension of the PEPS virtual spaces and $χ$ for the remaining virtual spaces of the environment.
"""

# ╔═╡ b1c0ca10-e70a-42f9-bb49-ecba7a4c0c4e
md"""
## Initialization

In the initialization phase, we want to start from a given `InfinitePEPS` and set up the appropriate tensors to begin the CTMRG algorithm.
For convenience, we also add a function to initialize a random `InfinitePEPS`.
"""

# ╔═╡ 4bb53195-71b9-42d9-b801-c5bf4a674c14
function initialize_peps(T, physicalspace, north_virtualspace, east_virtualspace = north_virtualspace)
	south_virtualspace = north_virtualspace'
	west_virtualspace = east_virtualspace'
    peps_tensor = randn(T, physicalspace ← north_virtualspace ⊗ east_virtualspace ⊗ south_virtualspace ⊗ west_virtualspace)
	return InfinitePEPS(peps_tensor)
end

# ╔═╡ ec6e77bb-0fc5-4730-85af-c37d15b2aeed
md"""
Then, we do the same thing for the environment tensors.
In practice, there are two different ways of initializing an environment.

The first and easiest to implement is to simply start from a completely random environment.
"""

# ╔═╡ ff1bbb0b-6ff8-4835-a28f-f6b9ef221faa
function initialize_random_environment(double_peps::InfiniteDoublePEPS, boundary_virtualspace)
	T = scalartype(double_peps.tensor)
	
	north_virtualspace = space(double_peps.tensor, 3)'
	east_virtualspace = space(double_peps.tensor, 4)'
	south_virtualspace = space(double_peps.tensor, 2)'
	west_virtualspace = space(double_peps.tensor, 1)'
	
	edge_north = randn(T, boundary_virtualspace ⊗ north_virtualspace ← boundary_virtualspace)
	edge_east = randn(T, boundary_virtualspace ⊗ east_virtualspace ← boundary_virtualspace)
	edge_south = randn(T, boundary_virtualspace ⊗ south_virtualspace ← boundary_virtualspace)
	edge_west = randn(T, boundary_virtualspace ⊗ west_virtualspace ← boundary_virtualspace)

	corner_northeast = randn(T, boundary_virtualspace ← boundary_virtualspace)
	corner_northwest = randn(T, boundary_virtualspace ← boundary_virtualspace)
	corner_southeast = randn(T, boundary_virtualspace ← boundary_virtualspace)
	corner_southwest = randn(T, boundary_virtualspace ← boundary_virtualspace)

	return CTMRGEnvironment(
		edge_north, edge_east, edge_south, edge_west,
		corner_northwest, corner_northeast, corner_southeast, corner_southwest
	)
end

# ╔═╡ bcb0c185-4a10-4f74-af7c-dc2a4db632ac
md"""
An alternative that does not require a $χ$ as input is found by taking a more physically motivated approach.
For an infinite system, we can consider the boundaries at infinity to be open, or in other words we can start from the vacuum.
To achieve this, we want to use a dimension $χ = 1$, and simply connect the virtual legs of the bra and ket together.
We can do this without writing dedicated contractions by starting from appropriate identity `TensorMap`s:
"""

# ╔═╡ f6343314-79d7-4980-b9b6-c1a396ce4297
function initialize_trivial_environment(peps)
	missing
end

# ╔═╡ 020b5d5f-cd1b-40a8-ae0e-4b8e3d15e594
md"""
## CTMRG steps

Next we get to the main body of the algorithm.
As mentioned, CTMRG is an iterative fixed-point method.
Therefore, we can set up a skeleton for the algorithm as follows:
"""

# ╔═╡ 3983afb7-585b-4e43-aab1-de4ce47d685a
md"""
### Expansion

The first part of each iteration consists of inserting one additional row or column in the network, containing a bra-ket tensor pair and two environment edges.
After growing the network like that, we can obtain updated corner and edge tensors by contracting the added row into the environment.
For example, here we absorb the west edge into the north-west corner, the bra-ket pair into the north edge and the east edge into the north-east corner:

![CTMRG north expansion](https://github.com/lkdvos/PEPSTutorial/blob/main/notebooks/assets/expansion.jpeg?raw=true)
"""

# ╔═╡ b9ae7480-9341-4293-b615-d4b353ca39ed
function ctmrg_expand(environment, peps)
	@tensor C_northwest[s; w1 w2] := environment.edge_west[s w2; n] * environment.corner_northwest[n; w1]
	@tensor E_north[w1 w2 s; e1 e2] := environment.edge_north[w1 n; e1] * peps.tensor[w2 s; n e2]
	@tensor C_northeast[e1 e2; s] := environment.corner_northeast[e1; n] * environment.edge_east[n e2; s]
	return C_northwest, E_north, C_northeast
end

# ╔═╡ f5dd768c-97b4-48cc-be0c-ed36ffad371d
md"""
### Projection

An important thing to note is that while we could simply merge some of the legs of the expanded environment tensors to obtain the same shape again, this would yield exponentially growing environment dimensions $χ$, quickly rendering this process untractable.
Therefore, the CTMRG algorithm procedes by identifying an appropriate subspace of these expanded spaces, characterized by isometries.
This is the heart of CTMRG: constructing a good isometry from an effective reduced density matrix / Gram matrix derived from the enlarged environment.
In particular, the environment of the bond we aim to truncate is described as
```math
M = L * R
```
and we wish to have a truncation of the following form that approximates this as close as possible:
```math
\tilde{M} = L * P_R * P_L * R
```

This is achieved with a truncated singular value decomposition, and we can summarize this as follows:

![CTMRG projectors](https://github.com/lkdvos/PEPSTutorial/blob/main/notebooks/assets/projection.jpeg?raw=true)
"""

# ╔═╡ 71526bec-030b-476c-a264-db69f858aeb7
function ctmrg_projectors(environments, peps; maxdim::Int = 10)
	@tensor contractcheck = true L[-1 -2; -3 -4] :=
        environments.edge_south[-1 3; 1] *
		environments.corner_southwest[1; 2] *
		environments.edge_west[2 4; 5] *
		peps.tensor[4 3; 6 -2] *
		environments.edge_west[5 7; 8] *
		peps.tensor[7 6; 9 -4] *
		environments.corner_northwest[8; 10] *
		environments.edge_north[10 9; -3]
	
	@tensor contractcheck = true R[-1 -2; -3 -4] :=
		environments.edge_north[-1 3; 1] *
		environments.corner_northeast[1; 2] *
		environments.edge_east[2 4; 5] *
		peps.tensor[-2 6; 3 4] *
		environments.edge_east[5 7; 8] *
		peps.tensor[-4 9; 6 7] *
		environments.corner_southeast[8; 10] *
		environments.edge_south[10 9; -3]

	M = L * R
	U, Σ, Vᴴ = svd_trunc(M; trunc = truncrank(maxdim))
	Σ_invsqrt = inv(sqrt(Σ))
	PR = R * Vᴴ' * Σ_invsqrt
	PL = Σ_invsqrt * U' * L

	return PR, PL
end

# ╔═╡ 1c6868d8-49aa-41a5-ac63-a09fa5e60fc4
md"""
### Renormalization

Finally, we can make use of these constructors to reduce the expanded tensors back to a manageable dimension.

![CTMRG projectors](https://github.com/lkdvos/PEPSTutorial/blob/main/notebooks/assets/renormalization.jpeg?raw=true)
"""

# ╔═╡ 0689c458-4352-4800-a7e2-29b9adfce426
function ctmrg_renormalize(C_northwest, E_north, C_northeast, PR, PL)
	C_northwest′ = C_northwest * PR
	@tensor E_north′[-1 -2; -3] := PL[-1; 3 4] * E_north[3 4 -2; 1 2] * PR[1 2; -3]
	C_northeast′ = PL * C_northeast
	return normalize(C_northwest′), normalize(E_north′), normalize(C_northeast′)
end

# ╔═╡ 0ba5f43a-509f-49e5-ae4d-7d4c995a60fc
md"""
Bringing it all together, we can perform a single direction of the CTMRG algorithm as follows:
"""

# ╔═╡ a5c48034-af8a-402f-bbf7-47c5d9aed52e
function ctmrg_north_itertion(environment, peps; kwargs...)
	C_northwest, E_north, C_northeast = ctmrg_expand(environment, peps)
	PR, PL = ctmrg_projectors(environment, peps; kwargs...)
	C_northwest′, E_north′, C_northeast′ = ctmrg_renormalize(C_northwest, E_north, C_northeast, PR, PL)
	return CTMRGEnvironment(E_north′, environment.edge_east, environment.edge_south, environment.edge_west, C_northwest′, C_northeast′, environment.corner_southeast, environment.corner_southwest)
end

# ╔═╡ c410317c-9fd6-4c05-8053-544f24534a0f
md"""
### Rotation

In principle we can now repeat this process in the exact same way for the 3 other cardinal directions.
However, this would be quite tedious, and instead we can simply rotate the environment as well as the state, and then re-use the methods we defined before.
"""

# ╔═╡ e4bda769-1aaf-4bcc-a2ca-600adb988639
function rotate_clockwise(peps::InfiniteDoublePEPS)
	@tensor peps′[S E; W N] := peps.tensor[W S; N E]
	return InfiniteDoublePEPS(peps′)
end

# ╔═╡ f95272cc-1b29-479f-a1ec-3128bf393204
function rotate_clockwise(environment::CTMRGEnvironment)
	return CTMRGEnvironment(
		environment.edge_west, environment.edge_north, environment.edge_east, environment.edge_south,
		environment.corner_southwest, environment.corner_northwest, environment.corner_northeast, environment.corner_southeast
	)
end

# ╔═╡ f7bd71f9-f785-452a-a606-84d0f823f09d
function ctmrg_iteration(environment, peps; kwargs...)
	for _ in 1:4
		environment = ctmrg_north_itertion(environment, peps; kwargs...)
		environment = rotate_clockwise(environment)
		peps = rotate_clockwise(peps)
	end
	return environment
end

# ╔═╡ ea38966c-c34d-4e5a-ad74-5cdc5d9c8c77
function ctmrg(environment, peps; maxiter = 100, kwargs...)
	@info "Starting CTMRG"

	for iter in 1:maxiter
		environment = ctmrg_iteration(environment, peps; kwargs...)
		@info "Iteration $iter" χ = dim(space(environment.corner_northeast, 1))
	end

	return environment
end

# ╔═╡ 3b69e064-185b-4c56-8e75-b062059f3f08
md"""
### Trying it out
"""

# ╔═╡ 31384227-6e84-4bf7-b603-be19b08638ce
let # this is a block to not leak any variables to other parts of the notebook
	peps = initialize_peps(Float64, ComplexSpace(2), ComplexSpace(3))
	double_peps = merge_braket(peps)
	environments = initialize_random_environment(double_peps, ComplexSpace(10))
	environments = ctmrg(environments, double_peps)
end

# ╔═╡ df668aea-2156-40f4-8dbd-2a442e863210


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
TensorKit = "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"

[compat]
TensorKit = "~0.16.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.3"
manifest_format = "2.0"
project_hash = "7c145bd57f6c98a0fcd7debbff373a88c604fe30"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "856ecd7cebb68e5fc87abecd2326ad59f0f911f3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.43"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "a49f9342fc60c2a2aaa4e0934f06755464fcf438"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.6"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.ChunkSplitters]]
git-tree-sha1 = "63a3903063d035260f0f6eab00f517471c5dc784"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.1.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.HalfIntegers]]
git-tree-sha1 = "9c3149243abb5bc0bad0431d6c4fcac0f4443c7c"
uuid = "f0d1745a-41c9-11e9-1dd9-e5d34d218721"
version = "1.6.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

    [deps.InverseFunctions.weakdeps]
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.LRUCache]]
git-tree-sha1 = "5519b95a490ff5fe629c4a7aa3b3dfc9160498b3"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.2"

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

    [deps.LRUCache.weakdeps]
    Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.MatrixAlgebraKit]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6f5785041912bf6200caa6dea26fc6b56e55cd58"
uuid = "6c742aac-3347-4629-af66-fc926824e5e4"
version = "0.6.1"

    [deps.MatrixAlgebraKit.extensions]
    MatrixAlgebraKitAMDGPUExt = "AMDGPU"
    MatrixAlgebraKitCUDAExt = "CUDA"
    MatrixAlgebraKitChainRulesCoreExt = "ChainRulesCore"
    MatrixAlgebraKitGenericLinearAlgebraExt = "GenericLinearAlgebra"
    MatrixAlgebraKitGenericSchurExt = "GenericSchur"
    MatrixAlgebraKitMooncakeExt = "Mooncake"

    [deps.MatrixAlgebraKit.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GenericLinearAlgebra = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
    GenericSchur = "c145ed77-6b09-5dd9-b285-bf645a82121e"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "ScopedValues", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "5ece5a3bbfe756517da7b9f1969a66f92fe62ad4"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.8.4"

    [deps.OhMyThreads.extensions]
    MarkdownExt = "Markdown"

    [deps.OhMyThreads.weakdeps]
    Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"

    [deps.PackageExtensionCompat.weakdeps]
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RationalRoots]]
git-tree-sha1 = "e5f5db699187a4810fda9181b34250deeedafd81"
uuid = "308eb6b3-cc68-5ff3-9e97-c3c4da4fa681"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.Strided]]
deps = ["LinearAlgebra", "StridedViews", "TupleTools"]
git-tree-sha1 = "c2e72c33ac8871d104901db736aecb36b223f10c"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "2.3.2"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "425158c52aa58d42593be6861befadf8b2541e9b"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.4.1"

    [deps.StridedViews.extensions]
    StridedViewsCUDAExt = "CUDA"
    StridedViewsPtrArraysExt = "PtrArrays"

    [deps.StridedViews.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    PtrArrays = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TaskLocalValues]]
git-tree-sha1 = "67e469338d9ce74fc578f7db1736a74d93a49eb8"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.3"

[[deps.TensorKit]]
deps = ["LRUCache", "LinearAlgebra", "MatrixAlgebraKit", "OhMyThreads", "Printf", "Random", "ScopedValues", "Strided", "TensorKitSectors", "TensorOperations", "TupleTools", "VectorInterface"]
git-tree-sha1 = "79e76c857beabc9c1d7f2a1b10d7f3e4463d5c77"
uuid = "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"
version = "0.16.0"

    [deps.TensorKit.extensions]
    TensorKitChainRulesCoreExt = "ChainRulesCore"
    TensorKitFiniteDifferencesExt = "FiniteDifferences"

    [deps.TensorKit.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"

[[deps.TensorKitSectors]]
deps = ["HalfIntegers", "LinearAlgebra", "TensorOperations", "WignerSymbols"]
git-tree-sha1 = "bb54ef851826493b1ec37fd6a9a9ec573ccd5373"
uuid = "13a9c161-d5da-41f0-bcbd-e1a08ae0647f"
version = "0.3.4"

[[deps.TensorOperations]]
deps = ["LRUCache", "LinearAlgebra", "PackageExtensionCompat", "PrecompileTools", "Preferences", "PtrArrays", "Strided", "StridedViews", "TupleTools", "VectorInterface"]
git-tree-sha1 = "874d1dfcb9f444c750928cf4e4556098c05f88c5"
uuid = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"
version = "5.3.1"

    [deps.TensorOperations.extensions]
    TensorOperationsBumperExt = "Bumper"
    TensorOperationsChainRulesCoreExt = "ChainRulesCore"
    TensorOperationscuTENSORExt = ["cuTENSOR", "CUDA"]

    [deps.TensorOperations.weakdeps]
    Bumper = "8ce10254-0962-460f-a3d8-1f77fea1446e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.VectorInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9166406dedd38c111a6574e9814be83d267f8aec"
uuid = "409d34a3-91d5-4945-b6ec-7529ddf182d8"
version = "0.5.0"

[[deps.WignerSymbols]]
deps = ["HalfIntegers", "LRUCache", "Primes", "RationalRoots"]
git-tree-sha1 = "960e5f708871c1d9a28a7f1dbcaf4e0ee34ee960"
uuid = "9f57e263-0b3d-5e2e-b1be-24f2bb48858b"
version = "2.0.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╠═7d26a11c-f98c-473a-9d11-9c564deb1272
# ╟─9b2b4b2e-eb14-11f0-b4d4-79004a9b7521
# ╟─44833ecc-202c-40bd-b761-dbd25c0f6f7b
# ╟─df08983e-dab0-4a00-bf0c-4eb91f42f1fa
# ╟─20f29190-2e0d-42e3-af82-11761478d29d
# ╟─ab8cc3a0-0fd4-4f11-be1d-40b2a44f53d2
# ╠═d2c17a80-1c4a-49be-b615-ed6eb53b0779
# ╠═a3a87f5e-da1a-45f2-b45b-8df484a30de8
# ╠═82fa0408-7690-41db-8ced-823aeb662974
# ╟─9da0bb7d-f918-40a1-871f-a546d7b5f641
# ╠═c0862aa4-715b-412a-97ec-5bdf00a5fbdf
# ╠═f2a65413-604c-45f4-99cc-e4f9abc914be
# ╟─4cb24d7c-e175-4cab-90f1-dc91a1d84ec1
# ╠═5d501f80-2e05-48ed-b62e-52ed723a2e90
# ╠═076f533f-1e61-4172-b118-d0815e8c66cd
# ╠═b2b002e3-5c08-4cc6-89c3-07870bf666d7
# ╟─7a94202f-3487-47c7-923a-61a178257374
# ╟─b1c0ca10-e70a-42f9-bb49-ecba7a4c0c4e
# ╠═4bb53195-71b9-42d9-b801-c5bf4a674c14
# ╟─ec6e77bb-0fc5-4730-85af-c37d15b2aeed
# ╠═ff1bbb0b-6ff8-4835-a28f-f6b9ef221faa
# ╟─bcb0c185-4a10-4f74-af7c-dc2a4db632ac
# ╟─f6343314-79d7-4980-b9b6-c1a396ce4297
# ╟─020b5d5f-cd1b-40a8-ae0e-4b8e3d15e594
# ╠═ea38966c-c34d-4e5a-ad74-5cdc5d9c8c77
# ╟─3983afb7-585b-4e43-aab1-de4ce47d685a
# ╠═b9ae7480-9341-4293-b615-d4b353ca39ed
# ╟─f5dd768c-97b4-48cc-be0c-ed36ffad371d
# ╠═71526bec-030b-476c-a264-db69f858aeb7
# ╟─1c6868d8-49aa-41a5-ac63-a09fa5e60fc4
# ╠═0689c458-4352-4800-a7e2-29b9adfce426
# ╟─0ba5f43a-509f-49e5-ae4d-7d4c995a60fc
# ╟─a5c48034-af8a-402f-bbf7-47c5d9aed52e
# ╟─c410317c-9fd6-4c05-8053-544f24534a0f
# ╠═e4bda769-1aaf-4bcc-a2ca-600adb988639
# ╟─f95272cc-1b29-479f-a1ec-3128bf393204
# ╠═f7bd71f9-f785-452a-a606-84d0f823f09d
# ╟─3b69e064-185b-4c56-8e75-b062059f3f08
# ╠═31384227-6e84-4bf7-b603-be19b08638ce
# ╠═df668aea-2156-40f4-8dbd-2a442e863210
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
