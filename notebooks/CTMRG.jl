### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 7d26a11c-f98c-473a-9d11-9c564deb1272
begin
	# setting up the notebook
	using PlutoUI
	using PlutoTeachingTools
end

# ╔═╡ 02cf1c6d-c588-4f49-8de0-4693ed33d21d
using TensorKit

# ╔═╡ 30e688b4-293f-40af-a839-31fabec981cc
using TensorKitTensors: SpinOperators

# ╔═╡ 7eb8edae-9d1b-4311-af50-6a258840f8eb
using OptimKit

# ╔═╡ d6e0fb57-f686-4400-aa92-748288e0591d
using Zygote

# ╔═╡ aeeae089-8fd3-4f85-b0bb-8773f9696073
TableOfContents()

# ╔═╡ 90e7d0b6-79f6-4587-85f9-5f1b68a79fdc
ReadingTimeEstimator()

# ╔═╡ 9b2b4b2e-eb14-11f0-b4d4-79004a9b7521
md"""
# CTMRG for infinite PEPS with TensorKit.jl

In this tutorial, we aim to write the code to contract infinite networks through the CTMRG algorithm, making use of TensorKit.jl.
This notebook contains an example setup for implementing this algorithm, where some parts are left as exercise to the reader, while others are filled in to speed up the process.
At the end of this tutorial, we want to be able to:

- contract the norm-network of an infinite PEPS $\langle\psi|\psi\rangle$
- compute expectation values of (local) observables for a given PEPS
- find a PEPS representation that minimizes $\langle \psi | H | \psi \rangle$

To accomplish this, we will leverage TensorKit.jl to provide the building blocks for creating the algorithm, and will first go over some basic aspects of that package.
Then, we aim to set up a basic CTMRG algorithm to compute the energy of the transverse-field Ising model on a 2D square lattice, and finally try a very naive approach to optimizing a ground state.

This notebook is structured to be followed as a guided exercise session, and the goal is to fill out the parts that are indicated as `missing`.
As the notebook aims to break down the required algorithm in different parts, various tests are written to automatically verify the correctness of the intermediate steps.
Note however that if you ever get stuck, you shouldn't hesitate to reach out and ask questions.
Additionally, since this notebook is fully self-contained, example solutions can be found in the appendix at the bottom of this page, which are also used to verify the intermediate results.

Don't hesitate to add cells to play around with various expressions and objects inbetween the rest of the code.
Often, there is no better or more intuitive explanation than simply inspecting some variables yourself.
However, keep in mind that unlike a typical Jupyter notebook, a Pluto notebook will not allow you to redefine variables with the same name, and additionally you are only allowed to define a single variable per cell.
This is what makes its reactive model work, and also means cells can be executed out of order.
Therefore, whenever you wish to inspect some local variables without affecting the remainder of the code, or reuse some variable names later, it can be useful to hide these from Pluto by using various scope constructions.
Typically, a `let ... end` block does the trick best.
"""

# ╔═╡ 043300a2-2333-4552-a9c1-d95dead19e51
let
	test_variable = 1 # this variable does not outlive this scope
end

# ╔═╡ 470aa499-b5b8-440b-bd48-095aac2377cf
test_variable = 1 # this variable does outlive this scope

# ╔═╡ 8b70a006-1f58-4181-ae67-f5dbb8a733a7
md"""
# I. TensorKit

We start off with a small primer on the use of TensorKit and its features.
TensorKit.jl provides a high-level approach to tensor network programming.
Rather than treating tensors as anonymous multidimensional arrays, TensorKit models them as linear maps between vector spaces, optionally equipped with symmetries.
While this perspective may seem somewhat pedantic, it is essential for writing symmetry-agnostic PEPS algorithms.

In this prologue, we introduce the core ideas needed for the rest of the tutorial:

- How TensorKit represents tensor indices,
- How to construct tensors with given domains and codomains,
- How to perform index manipulations and contractions,
- How to use factorizations (SVD, eigendecompositions, ...).

The goal is not to exhaust the TensorKit API, but to establish a mental model that will carry through the CTMRG notebook.
For further information, the first place to look is [the TensorKit documentation](https://quantumkithub.github.io/TensorKit.jl/stable/man/tutorial/), which provides more in-depth explanations of various parts of the library.
"""

# ╔═╡ 715a210a-7f7b-438a-8b08-b1494dbf99db
md"""
!!! note
	It can often be helpful to quickly look up the docstring of a function using the *Live docs* in the bottom right of this notebook.
	This can provide function signatures, examples and quick explanations of many parts of both TensorKit and Julia as a whole
"""

# ╔═╡ e995f859-91df-4a8b-9826-11a31c9fa667
md"""
## Tensor Indices

In TensorKit, a tensor index is not just an integer range `1:d`.
Instead, every index corresponds to a vector space which may have additional structure.
Various types of vector spaces can be defined in TensorKit, but here we will focus on the `ComplexSpace` type.
These represent complex vector spaces that are equipped with with a notion of duality, i.e. ``V' \neq V``.
For our purposes, this boils down to considering dressed tensor networks, where each edge is equipped with an arrow to assign a directionality.
Not only is this viewpoint unavoidable for more structured vector spaces (when using symmetries), it additionally tends to help catch silent bugs earlier, since non-matching arrows can't be contracted or added.
"""

# ╔═╡ e37310a8-1e24-4766-bb62-55d0154815bb
let # this block ensures variables do not escape
	V1 = ComplexSpace(2)
	V2 = ℂ^2 # \bbC + TAB
	V3 = V1'
	V = V1 ⊗ V2 ⊗ V3 # \otimes + TAB
end

# ╔═╡ 00b88ea4-3675-41a4-905e-a8103e23e4d6
md"""
Crucially, TensorKit additionally distinguishes between:

- Incoming indices (domain),
- Outgoing indices (codomain).

This makes tensors explicit linear maps rather than mere undirected arrays:

```math
T : V_1 \otimes V_2 \rightarrow W_1 \otimes W_2
```

This bipartition of the indices is essential for defining operations that are *matrix-like* in nature on a more general tensor.
As such, a `TensorMap` can typically be thought of as both a *vector* and a *matrix* at the same time, seamlessly transitioning between the two points of view whenever required.

As a consequence, often the lower-level operations are specified using two tuples of integers rather than a single one to indicate the bipartition of the result.
"""

# ╔═╡ e6325d88-0ae6-47e8-802f-293db7b7f560
let
	V = ℂ^2 ⊗ (ℂ^3)' ← ℂ^4 ⊗ ℂ^5
	permute(V, ((1, 2, 3), (4,)))
end

# ╔═╡ 391f207c-79c0-419e-adf4-b66de67e6d57
md"""
The attentive reader may have noticed that the duality of the third space of `V` and its permuted version above has changed.
This is not an accident, it stems from the following equivalence:

```math
\begin{align}
&V_1 \otimes V_2 \otimes \cdots \otimes V_N ≃\\
&V_1 \otimes V_2 \otimes \cdots \otimes V_{N_1} \leftarrow V_{N_1+1}^* \otimes \cdots \otimes V_N^*
\end{align}
```

In other words, dual spaces in the codomain and non-dual spaces in the domain correspond to *outgoing* arrows, while non-dual spaces in the codomain and dual spaces in the domain correspond to *incoming* arrows.
"""

# ╔═╡ 6af70e05-d625-40ee-ae29-d8b24379eb30
md"""
!!! note
	While the mathematical literature tends to define maps as `domain → codomain`, TensorKit tends to favor the `codomain ← domain` convention.
	Here, the arrows follow the direction of matrix multiplication, i.e. we have
	```math
	\begin{align}
	&A : V_3 \leftarrow V_2 \\
	&B : V_2 \leftarrow V_1 \\
	\implies &C = A \cdot B : V_3 \leftarrow V_1
	\end{align}
	```
	Note however that both syntaxes are supported, and can be accessed by typing `\leftarrow+TAB` or `\rightarrow+TAB`.
"""

# ╔═╡ b74fcb1a-6252-4350-b274-7b05b35dc9a8
md"""
## Tensor Construction

Once spaces are declared and a bipartition is chosen, tensors can be created.
This is often achieved by using Julia's standard library functions `Base.zeros`, `Base.rand` or `Base.randn`.
For details on how these functions, you can consider exploring the Live docs in the bottom right of this notebook and e.g. searching for `Base.zeros`.
You can access the tensor data using typical `t[i, j, ...]` slicing operations.
"""

# ╔═╡ cbfc4daf-c79c-4e2a-a70a-f5208cf31975
let
	t1 = rand(ℂ^2 ← ℂ^3)
	t2 = zeros(ℂ^2 ← ℂ^3)
	t3 = rand(ComplexF64, ℂ^2 ← ℂ^2)
end;

# ╔═╡ 439133a3-10c6-4da6-a671-38d2453d086b
md"""
Furthermore, various special-purpose constructors exist that can be useful for various linear algebra-oriented goals.
These tend to follow the same signature, and we refer to their docstrings for more information:

- `id`
- `isometry` and `isomorphism`
- `unitary`
"""

# ╔═╡ d05ac00c-1e69-4952-9629-80468f02096d
md"""
Finally, various often-used tensors are provided through the light-weight [TensorKitTensors.jl](https://github.com/QuantumKitHub/TensorKitTensors.jl) package.
Specifically when dealing with symmetries this is often the most convenient option.
If you find that your specific tensor is missing, don't hesitate to reach out or open an issue on the GitHub page!

Here we import the spin operators in order to later define the Hamiltonian of the transverse-field Ising model.
"""

# ╔═╡ d73e61f1-622e-49b2-97bb-dec56807d117
md"""
## Index Manipulation and Contractions

In tensor network algorithms, “index manipulation” tends to take up half the work: you are constantly reordering legs, contracting them, grouping them into composite indices, inserting identities or gauges, ...
To allow these operations, TensorKit.jl conveniently re-exports the functionality provided by [TensorOperations.jl](https://github.com/QuantumKitHub/TensorOperations.jl/).
In particular, the `@tensor` macro can be used to provide an intuitive [*Einstein notation*](https://en.wikipedia.org/wiki/Einstein_notation)-based syntax that is easier to read as well as performant.
Being a Julia macro, the index patterns are evaluated only once when the code is parsed, and baked into the compiled code afterwards.

There are three recurring patterns we will use throughout CTMRG and observable calculations:

1. Permuting legs (reordering indices to match a desired contraction pattern),
2. Contracting networks of tensors,
3. Fusing / splitting legs (turning multiple indices into one composite index and back).

Here we will show how the `@tensor` macro can be used to achieve each of these goals.
"""

# ╔═╡ f9a33166-f457-4cf9-9409-9917c23d51fe
md"""
!!! warning
	As we are now starting to combine different tensors in various ways, we have to be careful to make sure to only combine matching indices.
	In particular, we can only add tensors that have matching spaces (both the individual indices as well as the bipartition have to be equal), and we may only contract over indices that are equal and have matching arrows.
	Whenever disallowed combinations are attempted, TensorKit will throw a `SpaceMismatch` error.
"""

# ╔═╡ aee1fb88-0631-483f-a7fd-7ecf8c66f06b
md"""
### Permutations

Starting off with the reordering of indices, we can simply assign various symbols to the various indices of a tensor, and reorder them in the output.
For example, the following snippet would construct the transpose of an input matrix `A`:

```julia
@tensor Aᵀ[j; i] := A[i; j]
```

Various remarks can be made for this simple expression.
1. Any combination of symbols can be used as a name for one of the indices, including signed and unsigned integers, longer names such as `myindexname_123` as well as various *prime*-levels `i'`, `i''`, etc.
2. The `;` between the symbols on the left-hand side of the equation is used to specify the bipartition of the resulting `TensorMap`. The `;` between the symbols on the right-hand side of the equation can be omitted and is simply there for readability.
3. The `:=` symbol is used to indicate that we wish to construct a completely *new* output `TensorMap`. Replacing this with `+=`, `-=` or `=` instead attempts to update a pre-existing object.

With this, it is possible to create more advanced combinations of permuted input tensors, as well as linear combinations thereof.
For example:

```julia
@tensor B2[i j k l] := B1[l k j i] # equivalent to B2[i j k l; ()]
@tensor C[1; 2] := C1[1 2] + C2[1; 2] + C3[(); 1 2] # repartitioning the domain-codomain
# ...
```
"""

# ╔═╡ 9577bf9b-1ce2-4497-8de4-5b7ce5e86010
md"""
### Contractions

Repeated indices on the right-hand side of the equations are assumed to be contracted, i.e. summed over.
These repeated indices are allowed to appear on different tensors (*contraction*) or on the same tensor (*partial trace*), but can only appear twice, and not more.
Repeated indices on the left-hand side is not allowed.
Otherwise, the same rules as for the index permutations still apply.
For example:

```julia
@tensor D[a, d, j, i, g] := A[a, b, c, d, e] * B[b, e, f, g] * C[c, f, i, j]
```

For networks of tensors that involve more than two input tensors, the computation is carried out by pairwise contraction.
Even though the exact contraction order, i.e. which pairwise contractions to perform, has no influence on the values of the resulting tensor, it can have a drastic effect on the runtime.
By default, the `@tensor` macro will simply evaluate expressions in a left-to-right manner.
In other words, the previous expression is equivalent to `(A[a, b, c, d, e] * B[b, e, f, g]) * C[c, f, i, j]`.
However, parentheses within the `@tensor` expressions are respected, and can therefore be used to alter the default order.
Furthermore, whenever exclusively integers are used as labels, the contraction will take place by ascending order of the labels, i.e. according to the [NCON convention](https://arxiv.org/abs/1402.0939).
Finally, it can be useful to explicitly specify the order without altering the labels by adding a `order = (...)` keyword expression to the `@tensor` call.

```julia
A = rand(ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2)
B = rand((ℂ^2)' ⊗ (ℂ^2)' ⊗ ℂ^2 ⊗ ℂ^2)
C = rand((ℂ^2)' ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2)
@tensor order = (c, f, b, e) D[a, d, j, i, g] := A[a, b, c, d, e] * B[b, e, f, g] * C[c, f, i, j]
```

"""

# ╔═╡ 32944d9c-c5ad-48eb-a9b0-69ed0df03acf
md"""
Often determining the optimal contraction order is not an easy task, as it depends on the topology of the network, as well as on the sizes of the input tensors.
For these cases, it can be convenient to construct an example with tensors that represent real workloads, and verify at runtime that the contraction orders are correct.
For this purpose, the `costcheck = warn` keyword combination can be used, which will warn for suboptimal contraction orders at runtime and provide you with an optimal order to insert into the expression.
Keep in mind that this runtime check can be expensive though, and should typically be disabled whenever the optimal order has been identified.
"""

# ╔═╡ af4bac3d-02cc-48d5-ae3a-df31c7f3e82a
let
	A = rand(ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2)
	B = rand((ℂ^2)' ⊗ (ℂ^2)' ⊗ ℂ^2 ⊗ ℂ^2)
	C = rand((ℂ^2)' ⊗ (ℂ^2)' ⊗ ℂ^2 ⊗ ℂ^2)
	@tensor costcheck = warn order = (c, f, b, e) D[a, d, j, i, g] := A[a, b, c, d, e] * B[b, e, f, g] * C[c, f, i, j]
end;

# ╔═╡ b245f91b-df80-4a1b-a6f9-d1b3ceb018da
md"""
Finally, whenever a `SpaceMismatch` is thrown it can be a bit cumbersome to try and identify which index is responsible for this.
To that end, the `contractcheck = true` keyword can be added to insert runtime checks that provide more helpful messages, pinpointing the exact origin of the error to a specific symbol.

Turn on `contractcheck`: $(@bind do_contractcheck Switch())
"""

# ╔═╡ a4abe1b6-f97a-4d81-abc3-4e930dff7632
let
	A = rand(ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2)
	B = rand(ℂ^2 ⊗ (ℂ^2)' ⊗ ℂ^2 ⊗ ℂ^2)
	C = rand((ℂ^2)' ⊗ (ℂ^2)' ⊗ ℂ^2 ⊗ ℂ^2)
	if do_contractcheck
		@tensor contractcheck = true D[a, d, j, i, g] := A[a, b, c, d, e] * B[b, e, f, g] * C[c, f, i, j]
	else
		@tensor D[a, d, j, i, g] := A[a, b, c, d, e] * B[b, e, f, g] * C[c, f, i, j]
	end
end;

# ╔═╡ c5838a61-56f4-49f6-b7d7-e1d738aa51af
md"""
!!! note
	The curious reader may wish to inspect the generated code of the `@tensor expression`.
	Since this code is not meant for human-consumption it can be difficult to parse, but there are (rare) cases where it can be useful to explicitly inspect which code is being run.
	For this, we can use `@macroexpand @tensor expression`.
"""

# ╔═╡ 56715c2b-8283-4a83-af08-fc72953b47e7
md"""
### Combining and Splitting Indices

Finally we wish to demonstrate how to merge and split different tensor indices.
To do so, we take a step back and ask ourselves what we are trying to do in terms of mathematical operations.
For this, we consider a small network with two `TensorMap`s `A` and `B`:

```math
\begin{align}
A : W \leftarrow V_1 \otimes V_2 \\ 
B : V_1 \otimes V_2 \leftarrow Z
\end{align}
```

We now effectively wish to transform this network such that we group indices corresponding to ``V_1`` and ``V_2``, without altering the value of the contracted network, i.e. we aim to find ``\tilde{A} : W \leftarrow V`` and ``\tilde{B} : V \leftarrow Z`` such that ``V \simeq V_1 \otimes V_2`` and

```math
C = A \cdot B = \tilde{A} \cdot \tilde{B}
```

This can be achieved by inserting a fusing and splitting tensor ``F : V \leftarrow V_1 \otimes V_2`` and ``F^{-1} : V_1 \otimes V_2 \leftarrow V``, such that we obtain

```math
C = A \cdot B = A \cdot F^{-1} \cdot F \cdot B = (A \cdot F^{-1}) \cdot (F \cdot B) = \tilde{A} \cdot \tilde{B}
```

Translating this into code, we would get:
"""

# ╔═╡ 53dc3a3d-db99-4ad9-8889-3afcd5864d84
let
	W, V1, V2, Z = ℂ^2, ℂ^2, ℂ^2, ℂ^2
	A = rand(W ← V1 ⊗ V2)
	B = rand(V1 ⊗ V2 ← Z)
	
	F = isomorphism(fuse(V1 ⊗ V2) ← V1 ⊗ V2)
	Ã = A * F'
	B̃ = F * B
	
	A * B ≈ Ã * B̃
end

# ╔═╡ 85dc0714-2f66-4bac-975e-a8af43fb4cbc
md"""
!!! note
	In principle, any invertible fusing tensor ``F`` would yield equivalent results, and the particular chosen value is merely a change of basis.
	Therefore, it is often convenient to choose a `unitary` fusing tensor instead, as these can be efficiently inverted through the `adjoint` operation.
	In TensorKit, both `unitary` and `isomorphism` will map to a `TensorMap` whos matrix representation is simply the identity, so both are unitary and efficient to construct
"""

# ╔═╡ 0600518e-aef2-4f9c-ad0f-0f082c58f391
md"""
## Factorizations

The final piece of the TensorKit puzzle before we are ready to move on to CTMRG consists of factorizations.
If the previous section can be thought of as ways of combining multiple tensors into a single output tensor, here we aim to do the opposite and create multiple tensors from a single input tensor.
Indeed, for most factorizations contracting the factors is typically equal to the original tensor, or at least provides a good approximation thereof.
Of course, we typically want some additional properties for these tensors, which depend on the type of factorization that is chosen.
In TensorKit, these are provided by [MatrixAlgebraKit.jl](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl).

For the purpose of this notebook, the main factorization that is required is the [singular value decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition), which factorizes a matrix ``A = U \Sigma V^H`` such that both ``U`` and ``V^H`` are isometries, and ``\Sigma`` is a diagonal matrix with real non-negative entries.
While typically defined for matrices, we can trivially extend this definition for `TensorMap`s by using their interpretation as linear map from the domain to the codomain spaces.
This factorization is particularly useful since it additionally allows optimal low-rank approximations of the input by discarding the smallest singular values.

In MatrixAlgebraKit, this is accessible through the following:

`maxdim = ` $(@bind maxdim Slider(1:100; default=25, show_value=true))
"""

# ╔═╡ faa5861e-bd36-4503-89bf-ed56a190250f
let
	A = rand(ℂ^100, ℂ^100)
	U, S, Vᴴ = svd_trunc(A; trunc = truncrank(maxdim))
	space(U), space(S), space(Vᴴ)
end

# ╔═╡ 394f7270-12f5-4ec0-b7b1-e8e5cd8c6f4e
md"""
!!! warning
	Factorizing a `TensorMap` will use the **current** bipartition of the provided tensor as axis around which to factorize, so we have to make sure we provide input tensors that have been correctly permuted
"""

# ╔═╡ f176dc62-b1f5-447c-958f-b08a6ff2efd7
md"""
# II. CTMRG

Now that we are comfortable (maybe?) with working with TensorKit.jl, we are ready to put this in action in the context of PEPS.
In particular, we wish to start contracting various local observables for infinite 2D networks.
Given such a tensor network (e.g. the PEPS norm network), the exact contraction thereof is already intractable for moderately-sized finite PEPS, and clearly impossible for an infinitely repeating network.
Instead, CTMRG approximates the infinite environment around a local region by a finite set of tensors:

- Four **corner tensors**: `corner_northwest, corner_northeast, corner_southeast, corner_southwest`
- Four **edge tensors**: `edge_north, edge_east, edge_south, edge_west`

$(PlutoUI.LocalResource("./assets/ctmrg_contraction.jpeg"))

These tensors represent an effective boundary that *summarizes* the infinite remainder of the lattice.
Once converged, this environment can be contracted with and without local insertions of operators to compute:
- norms and expectation values,
- energy densities for local Hamiltonians,
- horizontal and vertical correlation functions via effective transfer operators.

In practice, CTMRG is an iterative fixed-point method:
- Each move **grows** the boundary by absorbing one layer of the bulk network,
- Then **compresses** it back to bond dimension `χ` using an isometry derived from an effective reduced density matrix (or equivalent construction),
- Repeats until the boundary tensors stop changing (up to normalization / gauge).

We will gradually build up the components required for this algorithm, which we can outline as follows:
"""

# ╔═╡ df08983e-dab0-4a00-bf0c-4eb91f42f1fa
md"""
## Conventions

In order to all more or less agree on the conventions, and to make testing the code easier, we will start by setting up some conventions for the various tensors.
"""

# ╔═╡ 20f29190-2e0d-42e3-af82-11761478d29d
"""
    const PEPSTensor{T, S}

Default type for PEPS tensors of element type `T` and space type `S`, with a single physical index, and 4 virtual indices, conventionally ordered as: ``P ← N ⊗ E ⊗ S ⊗ W``.
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
Note however that this might not be optimal for performance, as the memory footprint of this tensor scales as ``D^8``.
"""

# ╔═╡ d2c17a80-1c4a-49be-b615-ed6eb53b0779
"""
    const DoublePEPSTensor{T, S}

Default type for double-layer PEPS tensors of element type `T` and space type `S`, with 4 virtual indices, conventionally ordered as: ``W ⊗ S ← N ⊗ E``.
Here, ``N``, ``E``, ``S`` and ``W`` denote the north, east, south and west doubled virtual spaces, respectively.

```
           N
          ╱
         ╱
   W---- ----E
       ╱ 
      ╱  
     S   
```
"""
const DoublePEPSTensor{T, S} = AbstractTensorMap{T, S, 2, 2}

# ╔═╡ a3a87f5e-da1a-45f2-b45b-8df484a30de8
"""
	merge_braket(peps::PEPSTensor)::DoublePEPSTensor

Contract a bra and ket tensor over the physical index, while merging the virtual spaces together.
See also the section on Combining and Splitting Indices.
"""
function merge_braket(peps::PEPSTensor)
	V_north = space(peps, 2) ⊗ space(peps, 2)'
	fuse_north = isomorphism(fuse(V_north) ← V_north)
	V_east = space(peps, 3) ⊗ space(peps, 3)'
	fuse_east = isomorphism(fuse(V_east) ← V_east)
	@tensor braket[S W; N E] := conj(peps[p; n' e' s' w']) * peps[p; n e s w] *
		fuse_north[N; n n'] * fuse_east[E; e e'] * conj(fuse_north[S; s s']) * conj(fuse_east[W; w w'])
	return braket
end

# ╔═╡ 9da0bb7d-f918-40a1-871f-a546d7b5f641
md"""
As a result, we can define the CTMRG environment tensors immediately for this double-layer object, leading to a 3-index tensor for the edges, and a 2-index tensor for the corners (as opposed to a 4-index tensor for the edges if we work with both layers separately).
"""

# ╔═╡ c0862aa4-715b-412a-97ec-5bdf00a5fbdf
"""
    const EdgeTensor{T, S}

Default type for double-layer CTMRG edge environment tensors of element type `T` and space type `S`, with indices conventionally ordered as: ``V_l ⊗ P ← V_r``.
Here, ``V_l`` and ``V_r`` denote the left and right virtual environment spaces and ``P`` denotes the PEPS virtual space.

```
V_l---- ----V_r
      ╱ 
     ╱  
    P   
```
"""
const EdgeTensor{T, S} = AbstractTensorMap{T, S, 2, 1}

# ╔═╡ f2a65413-604c-45f4-99cc-e4f9abc914be
"""
    const CornerTensor{T, S}

Default type for double-layer CTMRG corner environment tensors of element type `T` and space type `S`, with indices conventionally ordered as: ``V_l ← V_r``.
Here, ``V_l`` and ``V_r`` denote the left and right virtual environment spaces.

```
V_l --- --- V_r
```
"""
const CornerTensor{T, S} = AbstractTensorMap{T, S, 1, 1}

# ╔═╡ 4cb24d7c-e175-4cab-90f1-dc91a1d84ec1
md"""
Using this, we can set up the datastructures that we will use as follows:
"""

# ╔═╡ 5d501f80-2e05-48ed-b62e-52ed723a2e90
"""
	InfinitePEPS{A <: PEPSTensor}

Struct to represent an infinite PEPS built from a uniform grid of a single `PEPSTensor` repeated indefinitely.
"""
struct InfinitePEPS{A <: PEPSTensor}
    tensor::A
end

# ╔═╡ 076f533f-1e61-4172-b118-d0815e8c66cd
"""
	InfiniteDoublePEPS{A <: DoublePEPSTensor}

Struct to represent the combined double layer infinite PEPS built from a uniform grid of a single `DoublePEPSTensor` repeated indefinitely.
"""
struct InfiniteDoublePEPS{A <: DoublePEPSTensor}
    tensor::A
end

# ╔═╡ b2b002e3-5c08-4cc6-89c3-07870bf666d7
"""
	CTMRGEnvironment{E <: EdgeTensor, C <: CornerTensor}

Struct to represent the CTMRG environment tensors for a uniform double-layer infinite network.
"""
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
Throughout this tutorial, we will often use $d$ to refer to the dimension of the physical space, $D$ for dimension of the PEPS virtual spaces and $χ$ for the remaining virtual spaces of the environment.
"""

# ╔═╡ 1f0ba404-8001-4bc3-9132-1ce4a3a2e78d
function maximal_chi(environment::CTMRGEnvironment)
	return maximum(x -> dim(space(x, 1)), (environment.corner_northwest, environment.corner_northeast, environment.corner_southeast, environment.corner_southwest))
end

# ╔═╡ b1c0ca10-e70a-42f9-bb49-ecba7a4c0c4e
md"""
## Initialization

In the initialization phase, we want to start from a given `InfinitePEPS` and set up the appropriate tensors to begin the CTMRG algorithm.
For convenience, we also add a function to initialize a random `InfinitePEPS`.
"""

# ╔═╡ 4bb53195-71b9-42d9-b801-c5bf4a674c14
"""
	initialize_peps(T, physicalspace, north_virtualspace, east_virtualspace = north_virtualspace)::InfinitePEPS

Initialize a pseudo-random infinite PEPS state for a given element type `T <: Type{<:Number}`, `physicalspace`, `north_virtualspace` and `east_virtualspace`.
"""
function initialize_peps(T, physicalspace, north_virtualspace, east_virtualspace = north_virtualspace)
	south_virtualspace = north_virtualspace'
	west_virtualspace = east_virtualspace'
    peps_tensor = randn(T, physicalspace ← north_virtualspace ⊗ east_virtualspace ⊗ south_virtualspace ⊗ west_virtualspace)
	return peps_tensor
end

# ╔═╡ ec6e77bb-0fc5-4730-85af-c37d15b2aeed
md"""
Then, we do the same thing for the environment tensors.
The easiest way to implement this is to simply start from a completely random environment.
"""

# ╔═╡ ff1bbb0b-6ff8-4835-a28f-f6b9ef221faa
"""
	initialize_random_environment(double_peps::InfiniteDoublePEPS, boundary_virtualspace)::CTMRGEnvironment

Randomly initialize the environment tensors for a given double-layer PEPS and environment bond dimension.
"""
function initialize_random_environment(double_peps::DoublePEPSTensor, boundary_virtualspace)
	T = scalartype(double_peps)
	
	north_virtualspace = space(double_peps, 3)'
	east_virtualspace = space(double_peps, 4)'
	south_virtualspace = space(double_peps, 2)'
	west_virtualspace = space(double_peps, 1)'
	
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

# ╔═╡ 3983afb7-585b-4e43-aab1-de4ce47d685a
md"""
## Expansion

Next we get to the main body of the algorithm.
The first part of each iteration consists of inserting one additional row or column in the network, containing a bra-ket tensor pair and two environment edges.
After growing the network like that, we can obtain updated corner and edge tensors by contracting the added row into the environment.
For example, here we absorb the west edge into the north-west corner, the bra-ket pair into the north edge and the east edge into the north-east corner:

$(PlutoUI.LocalResource("./assets/expansion.jpeg"))
"""

# ╔═╡ effb78fe-894f-4b44-95ea-f3d80134bc75
"""
	ctmrg_expand(environment::CTMRGEnvironment, peps::DublePEPSTensor) -> C_northwest′, E_north′, C_northeast′

Compute the expanded north corners and edge of the CTMRG environment by inserting a row into the network, and contracting the resulting tensors.
These tensors should have the following shapes:
- `C_northwest′`: ``V_χ ← V_χ ⊗ V_D``
- `E_north′`: ``V_χ ⊗ V_D ⊗ V_D ← V_χ ⊗ V_D``
- `C_northeast′`: ``V_χ ⊗ V_D ← V_χ``
"""
function ctmrg_expand(environment::CTMRGEnvironment, peps::DoublePEPSTensor)
	missing
end

# ╔═╡ f5dd768c-97b4-48cc-be0c-ed36ffad371d
md"""
## Projection

An important thing to note is that while we could simply merge some of the legs of the expanded environment tensors to obtain the same shape again, this would yield exponentially growing environment dimensions $χ$, quickly rendering this process untractable.
Therefore, the CTMRG algorithm procedes by identifying an appropriate subspace of these expanded spaces, characterized by isometries.
This is the heart of CTMRG: constructing a good isometry from an effective reduced density matrix derived from the enlarged environment.
In particular, the environment of the bond we aim to truncate is described as
```math
M = L * R
```
and we wish to have a truncation of the following form that approximates this as close as possible:
```math
\tilde{M} = L * P_R * P_L * R
```

This is achieved with a truncated singular value decomposition, and we can summarize this as follows:


$(PlutoUI.LocalResource("./assets/projection.jpeg"))
"""

# ╔═╡ 607be339-8a6a-4a18-bbde-7775a349bbc1
"""
	ctmrg_projectors(environments::CTMRGEnvironments, peps::DoublePEPSTensor; maxdim::Integer = 10) -> PR, PL

Compute the appropriate CTMRG projectors of a given maximal dimension by first contracting the left- and right half `L` and `R` of a 4x4 grid, and then performing a truncated singular value decomposition on the result.
It can be convenient to choose the bipartition and indexorder of `L` and `R` such that no additional tensor contractions are required, and only matrix multiplications are involved.
The shape of the output tensors should be:
- `PR`: ``V_χ ⊗ V_D ← V_χ′``
- `PL`: ``V_χ′ ← V_χ ⊗ V_D``
"""
function ctmrg_projectors(environments, peps; maxdim::Integer = 10)
	missing
end

# ╔═╡ 1c6868d8-49aa-41a5-ac63-a09fa5e60fc4
md"""
## Renormalization

Finally, we can make use of these constructors to reduce the expanded tensors back to a manageable dimension.
To improve the numerical stability of our algorithm, we will choose to normalize the updated tensors such that they each have Frobenius norm `1`.
For this, see also `normalize`.

$(PlutoUI.LocalResource("./assets/renormalization.jpeg"))
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

# ╔═╡ d7579195-063f-49a5-95b8-8e88dc823321
"""
	ctmrg_north_iteration(environment, peps; kwargs...) -> environment

Expand the CTMRG environment along the north direction and return the updated, normalized environment after absorbing and renormalizing the additional row.
"""
function ctmrg_north_iteration(environment::CTMRGEnvironment, peps::DoublePEPSTensor; kwargs...)
	C_northwest, E_north, C_northeast = ctmrg_expand(environment, peps)
	PR, PL = ctmrg_projectors(environment, peps; kwargs...)
	C_northwest′, E_north′, C_northeast′ = ctmrg_renormalize(
		C_northwest, E_north, C_northeast, PR, PL
	)
	return CTMRGEnvironment(
		E_north′, environment.edge_east, environment.edge_south, environment.edge_west,
		C_northwest′, C_northeast′, environment.corner_southeast, environment.corner_southwest
	)
end

# ╔═╡ c410317c-9fd6-4c05-8053-544f24534a0f
md"""
## Rotation

In principle we can now repeat this process in the exact same way for the three other directions.
However, this would be quite tedious, and instead we can simply rotate the environment as well as the state, and then re-use the methods we defined before.
"""

# ╔═╡ e4bda769-1aaf-4bcc-a2ca-600adb988639
function rotate_clockwise(peps::DoublePEPSTensor)
	missing
end

# ╔═╡ f95272cc-1b29-479f-a1ec-3128bf393204
function rotate_clockwise(environment::CTMRGEnvironment)
	missing
end

# ╔═╡ 05e524c3-0369-4bea-8aea-eec522953d5f
md"""
Finally, we are ready to put all the pieces together and define the `ctmrg_iteration` function.
"""

# ╔═╡ f7bd71f9-f785-452a-a606-84d0f823f09d
"""
	ctmrg_iteration(environment::CTMRGEnvironment, peps::DoublePEPSTensor; kwargs...) -> environment

Compute the updated `CTMRGEnvironment`s obtained by expanding north followed by a rotation, repeated for each of the four directions.
"""
function ctmrg_iteration(environment::CTMRGEnvironment, peps::DoublePEPSTensor; kwargs...)
	missing
end

# ╔═╡ 3d4a07e4-f5c1-42d7-bc4d-a15a2fb36b66
md"""
## Convergence

Since CTMRG is an iterative method, we expect to have to repeat the CTMRG steps until at some point we decide that the environments have sufficiently converged.
However, we have to be slightly careful with our definition of convergence, since simply checking that two subsequent iterations have identical entries for all tensors runs into some problems.
This is a result of the gauge degrees of freedom that are left in the fixed-point equation of the environment: inserting ``X X^{-1}`` at any of the environment virtual legs does not alter the fixed point, but would affect element-wise convergence.
Therefore, we instead consider a measure of convergence that is not affected by these transformations, which is the difference of the singular value spectrum of the tensors.
"""

# ╔═╡ 8c32a473-4f08-4955-bdea-25cef4490de0
"""
	singular_value_distance(S1, S2)

Given the singular value spectrum of two objects, compute their distance by taking the norm of their difference.
Be careful, since the sizes of the inputs might not necessarily match, in which case missing entries should be treated as zeros.
"""
function singular_value_distance(S1, S2)
	missing
end

# ╔═╡ 2481bcb0-3c48-4d3e-97d4-5f00efd8ef2f
function ctmrg_convergence(env1, env2)
	return maximum(fieldnames(CTMRGEnvironment)) do f
		t1 = getproperty(env1, f)
		t2 = getproperty(env2, f)
		return singular_value_distance(svd_vals(t1), svd_vals(t2))
	end
end

# ╔═╡ ea38966c-c34d-4e5a-ad74-5cdc5d9c8c77
function ctmrg(environment, peps; maxiter = 100, tol = 1e-6, kwargs...)
	@info "Starting CTMRG"

	for iter in 1:maxiter
		environment_new = ctmrg_iteration(environment, peps; kwargs...)
		χ = maximal_chi(environment_new)
		svd_distance = ctmrg_convergence(environment, environment_new)
		@info "Iteration $iter" χ svd_distance
		environment = environment_new
		svd_distance <= tol && break
	end

	return environment
end

# ╔═╡ df668aea-2156-40f4-8dbd-2a442e863210
md"""
# III. Observables

Now that we have a way of approximately contracting an infinite PEPS network, we can look into computing local observables.
This is achieved by replacing the infinite environment around a local patch of tensors with the CTMRG approximation, and then contracting the remaining network.

!!! to do insert figure

However, it can be more convenient to work with reduced density matrices as an intermediate step in the calculation, which tends to avoid some duplicate work in contracting parts of the networks.
The reduced density matrix for some number of local sites can be defined as the double-layer norm network where we have left the physical indices of these sites uncontracted.
This can then be used to compute observables through the following identification:

```math
\frac{\langle \psi | O | \psi \rangle}{\langle \psi | \psi \rangle} = 
	\frac{\text{tr}(O | \psi \rangle \langle \psi |)}{\text{tr}(| \psi \rangle \langle \psi |)} = \frac{\text{tr}(O \cdot \rho)}{\text{tr}(\rho)}
```

!!! to do insert figure
"""

# ╔═╡ 67503430-d5d1-4ee6-847f-4b7dbecdf7ed
md"""
!!! note
	Since we have decided to merge the double-layer network when computing the CTMRG environment, we have to take a similar approach here as well.
	Therefore, we first make local fused density-matrix tensors, and then contract these into the environment
"""

# ╔═╡ 80907dcf-e9b6-47e8-89e9-f9e69a23ee63
function merge_ketbra(peps::InfinitePEPS)
	V_north = space(peps, 2) ⊗ space(peps, 2)'
	fuse_north = isomorphism(fuse(V_north) ← V_north)
	V_east = space(peps, 3) ⊗ space(peps, 3)'
	fuse_east = isomorphism(fuse(V_east) ← V_east)
	@tensor braket[P P'; N E S W] := conj(peps[P'; n' e' s' w']) * peps[P; n e s w] *
		fuse_north[N; n n'] * fuse_east[E; e e'] * conj(fuse_north[S; s s']) * conj(fuse_east[W; w w'])
end

# ╔═╡ f6acc633-d9ab-4cf5-b475-f608d0df13b2
function reduced_densitymatrix_1x1(peps, environment)
	ρ_local = merge_ketbra(peps)
	return @tensor ρ[-1; -2] :=
		environment.edge_west[11 3; 1] *
		environment.corner_northwest[1; 2] *
		environment.edge_north[2 4; 5] *
		environment.corner_northeast[5; 6] *
		environment.edge_east[6 7; 8] *
		environment.corner_southeast[8; 9] *
		environment.edge_south[9 10; 11] *
		ρ_local[-1 -2; 4 7 10 3]
end

# ╔═╡ d007a99a-fb6a-4b76-9edc-949fa2787b8d
function reduced_densitymatrix_1x2(peps, environment)
	ρ_local = merge_ketbra(peps, peps)
	return @tensor ρ[-1 -2; -3 -4] :=
		environment.edge_south[17 4; 1] *
		environment.corner_southwest[1; 2] *
		environment.edge_west[2 3; 5] *
		environment.corner_northwest[5; 6] *
		environment.edge_north[6 7; 15] *
		ρ_local[-1 -3; 7 16 4 3] *
		environment.edge_north[15 11; 8] *
		environment.corner_northeast[8; 9] *
		environment.edge_east[9 10; 12] *
		environment.corner_southeast[12; 13] *
		environment.edge_south[13 14; 17] *
		ρ_local[-2 -4; 11 10 14 16]
end

# ╔═╡ d97aae83-b513-4332-bc01-17fdd38b1381
function reduced_densitymatrix_2x1(peps, environment)
	peps_rotated = rotate_clockwise(peps)
	environment_rotated = rotate_clockwise(peps, environment)
	return reduced_densitymatrix_1x2(peps_rotated, environment_rotated)
end

# ╔═╡ f657e893-c238-419a-941a-f0ed9374e838
function expectation_value_1x1(O, peps, environment)
	ρ = reduced_densitymatrix_1x1(peps, environment)
	return tr(O * ρ) / tr(ρ)
end

# ╔═╡ 1c8c0edc-4ea6-470a-8b6c-ad4e9e2cb230
function expectation_value_1x2(O, peps, environment)
	ρ = reduced_densitymatrix_1x2(peps, environment)
	return tr(O * ρ) / tr(ρ)
end

# ╔═╡ 085669a2-0c44-4df2-93ce-475be4e45579
function expectation_value_2x1(O, peps, environment)
	ρ = reduced_densitymatrix_1x1(peps, environment)
	return tr(O * ρ) / tr(ρ)
end

# ╔═╡ 06bb1f23-a4cb-4b10-9d4e-b8f59c2171cc
md"""
# IV. Optimization

The final piece of the puzzle simply puts all pieces of the puzzle together.
We can use any black-box optimization routine we like, and here we will opt to use OptimKit.jl's LBFGS.
For this to work, on the one hand we need to define our optimization landscape by specifying an inner product and a retraction (how to take a step on the manifold), while on the other we need to tell which we want to go by providing a cost function and a gradient.
"""

# ╔═╡ cb153168-3a38-42b1-a2e5-4b5cd1062e49
md"""
## Optimization Manifold
Since we plan on making use of automatic differentiation to pr
"""

# ╔═╡ e58ed93c-4b37-44c4-8269-bb8cb039b4fa
md"""
## Gradient and Loss Function
"""

# ╔═╡ 209aa00c-0281-4e8c-b32c-d77888e8a5ae
md"""
# V. Conclusions

If everything went well, we should now be in a position to confidently start optimizing infinite PEPS for various Hamiltonians defined on a square lattice.
However, in all honesty we will quite quickly reach the limits of what our naive implimentation can achieve.
We will briefly discuss some of these short-comings here, some of them simply extensions that we could fix by investing a bit more time, and others quite complex algorithmic improvements that are less straightforward to achieve.

## Missing Features

## Missing Optimizations

## Further Research



## Open-source Libraries

While tackling some of these challenges can be a fun exercise, often this time can be better spent in a larger team, where forces can be bundled and progress can be shared.
In particular, we can refer to various of the presentations given this week for hints of what is currently being developed, and additionally the main message should be to go and check-out higher-level libraries.
Not only do these open-source libraries typically offer quite a few of these features out of the box, additionally quite a lot of time has gone into their design choices and optimizations.
Keep in mind that as they are open-source, they often welcome new users as well as eager contributors with open arms!

- PEPSKit.jl
- YASTN
- VariPEPS
- ...
"""

# ╔═╡ b333c5f9-ba3d-429c-9403-37d1f9fb7360
md"""
# Appendix
"""

# ╔═╡ dececc2a-5b73-4ad8-8056-e5c0a97a230f
md"""
## Solutions
"""

# ╔═╡ f897eafd-d376-4ed9-811b-76290bf11280
test_peps = rand(Float64, ComplexSpace(2) ← ComplexSpace(3) ⊗ ComplexSpace(3) ⊗ ComplexSpace(3)' ⊗ ComplexSpace(3)')

# ╔═╡ 14f49304-2286-48aa-b1b7-07a2e6f03cfe
test_doublepeps = merge_braket(test_peps)

# ╔═╡ f560e045-433f-4562-a611-119ce410342f
test_environment = initialize_random_environment(test_doublepeps, ComplexSpace(4))

# ╔═╡ b9ae7480-9341-4293-b615-d4b353ca39ed
function ctmrg_expand_solution(environment, peps::DoublePEPSTensor)
	@tensor C_northwest[s; w1 w2] := environment.edge_west[s w2; n] * environment.corner_northwest[n; w1]
	@tensor E_north[w1 w2 s; e1 e2] := environment.edge_north[w1 n; e1] * peps[w2 s; n e2]
	@tensor C_northeast[e1 e2; s] := environment.corner_northeast[e1; n] * environment.edge_east[n e2; s]
	return C_northwest, E_north, C_northeast
end

# ╔═╡ d6a71ccd-5bee-4532-bf29-03348ac7bb8e
let
	test_result = ctmrg_expand(test_environment, test_doublepeps)
	actual_result = ctmrg_expand_solution(test_environment, test_doublepeps)
	if ismissing(test_result)
		still_missing()
	elseif typeof(test_result) !== typeof(actual_result)
		keep_working(md"The type of the output is not correct. Did you return the required outputs?")
	else
		C_northwest, E_north, C_northeast = test_result
		C_northwest′, E_north′, C_northeast′ = actual_result
		if (space(C_northwest) != space(C_northwest′)) || (space(E_north) != space(E_north′)) || (space(C_northeast) != space(C_northeast′))
			almost(md"The spaces of (one of) the output tensors are not correct. Make sure to put the indices in the correct order and take the proper truncation dimension in mind.")
		elseif (C_northwest ≈ C_northwest′) && (E_north ≈ E_north′) && (C_northeast ≈ C_northeast′)
			correct()
		else
			keep_working()
		end
	end
end

# ╔═╡ 71526bec-030b-476c-a264-db69f858aeb7
function ctmrg_projectors_solution(environments, peps::DoublePEPSTensor; maxdim::Int = 10)
	@tensor L[-1 -2; -3 -4] :=
        environments.edge_south[-1 3; 1] *
		environments.corner_southwest[1; 2] *
		environments.edge_west[2 4; 5] *
		peps[4 3; 6 -2] *
		environments.edge_west[5 7; 8] *
		peps[7 6; 9 -4] *
		environments.corner_northwest[8; 10] *
		environments.edge_north[10 9; -3]
	
	@tensor R[-1 -2; -3 -4] :=
		environments.edge_north[-1 3; 1] *
		environments.corner_northeast[1; 2] *
		environments.edge_east[2 4; 5] *
		peps[-2 6; 3 4] *
		environments.edge_east[5 7; 8] *
		peps[-4 9; 6 7] *
		environments.corner_southeast[8; 10] *
		environments.edge_south[10 9; -3]

	M = L * R
	U, Σ, Vᴴ = svd_trunc(M; trunc = truncrank(maxdim))
	Σ_invsqrt = inv(sqrt(Σ))
	PR = R * Vᴴ' * Σ_invsqrt
	PL = Σ_invsqrt * U' * L

	return PR, PL
end

# ╔═╡ 941cfb90-087e-48ea-93df-8e72aaaaff8b
let
	test_result = ctmrg_projectors(test_environment, test_doublepeps; maxdim = 10)
	actual_result = ctmrg_projectors_solution(test_environment, test_doublepeps; maxdim = 10)
	if ismissing(test_result)
		still_missing()
	elseif typeof(test_result) !== typeof(actual_result)
		keep_working(md"The type of the output is not correct. Did you return the required outputs?")
	else
		PR, PL = test_result
		PR′, PL′ = actual_result
		if (space(PR) != space(PR′)) || (space(PL) != space(PL′))
			almost(md"The spaces of (one of) the output tensors are not correct. Make sure to put the indices in the correct order and take the proper truncation dimension in mind.")
		elseif (PR ≈ PR′) && (PL ≈ PL′)
			correct()
		else
			keep_working()
		end
	end
end

# ╔═╡ 99fad623-4b57-4aad-bc3a-66230fb81034
function ctmrg_renormalize_solution(C_northwest, E_north, C_northeast, PR, PL)
	C_northwest′ = C_northwest * PR
	@tensor E_north′[-1 -2; -3] := PL[-1; 3 4] * E_north[3 4 -2; 1 2] * PR[1 2; -3]
	C_northeast′ = PL * C_northeast
	return normalize(C_northwest′), normalize(E_north′), normalize(C_northeast′)
end

# ╔═╡ 49d2008e-8030-4c5f-8f98-cb2c42fd5d7d
function ctmrg_north_iteration_solution(environment::CTMRGEnvironment, peps::DoublePEPSTensor; kwargs...)
	C_northwest, E_north, C_northeast = ctmrg_expand_solution(environment, peps)
	PR, PL = ctmrg_projectors_solution(environment, peps; kwargs...)
	C_northwest′, E_north′, C_northeast′ = ctmrg_renormalize_solution(
		C_northwest, E_north, C_northeast, PR, PL
	)
	return CTMRGEnvironment(
		E_north′, environment.edge_east, environment.edge_south, environment.edge_west,
		C_northwest′, C_northeast′, environment.corner_southeast, environment.corner_southwest
	)
end

# ╔═╡ ada2cab6-fc02-4b05-a7d5-edbbae746dde
function rotate_clockwise_solution(peps::DoublePEPSTensor)
	return @tensor peps′[S E; W N] := peps[W S; N E]
end

# ╔═╡ 48270af5-c36a-499d-b0c1-f688820e7b84
function rotate_clockwise_solution(environment::CTMRGEnvironment)
	return CTMRGEnvironment(
		environment.edge_west, environment.edge_north, environment.edge_east, environment.edge_south,
		environment.corner_southwest, environment.corner_northwest, environment.corner_northeast, environment.corner_southeast
	)
end

# ╔═╡ b986149d-055a-4588-a08f-e2a86e803a38
let
	test_result = rotate_clockwise(test_doublepeps)
	actual_result = rotate_clockwise_solution(test_doublepeps)
	if ismissing(test_result)
		still_missing()
	elseif typeof(test_result) !== typeof(actual_result)
		keep_working(md"The type of the output is not correct. Did you return the required outputs?")
	else
		if (space(test_result) == space(actual_result)) && (test_result ≈ actual_result)
			correct()
		else
			almost(md"The returned value is incorrect. Did you rotate clockwise, or forgot the bipartition?")
		end
	end
end

# ╔═╡ 2506dde2-acac-41c9-935e-023b87223a8f
let
	test_result = rotate_clockwise(test_doublepeps)
	actual_result = rotate_clockwise_solution(test_doublepeps)
	if ismissing(test_result)
		still_missing()
	elseif typeof(test_result) !== typeof(actual_result)
		keep_working(md"The type of the output is not correct. Did you return the required outputs?")
	else
		if (all(fieldnames(CTMRGEnvironment)) do f
			t1 = getproperty(test_result, f)
			t2 = getproperty(actual_result, f)
			return (space(t1) == space(t2)) && (t1 ≈ t2)
		end)
			correct()
		else
			almost(md"The returned value is incorrect. Did you rotate clockwise, or forgot the bipartition?")
		end
	end
end

# ╔═╡ 3411519b-ac28-4983-b477-026c8d015b97
function ctmrg_iteration_solution(environment::CTMRGEnvironment, peps::DoublePEPSTensor; kwargs...)
	for direction in 1:4
		# one iteration in north direction
		environment = ctmrg_north_iteration_solution(environment, peps; kwargs...)

		# one rotation
		environment = rotate_clockwise_solution(environment)
		peps = rotate_clockwise_solution(peps)
	end
	return environment
end

# ╔═╡ 80afa573-0cc0-4c7e-98d6-f58bdf9205d9
function singular_value_distance_solution(S1, S2)
	S1_extended = vcat(S1, zeros(max(0, length(S2) - length(S1))))
	S2_extended = vcat(S2, zeros(max(0, length(S1) - length(S2))))
	return norm(S1_extended - S2_extended)
end

# ╔═╡ 0d6f69d6-7ed0-487a-852b-241d2580e1f3
let
	test_result = ctmrg_iteration(test_environment, test_doublepeps)
	actual_result = ctmrg_iteration_solution(test_environment, test_doublepeps)
	if ismissing(test_result)
		still_missing()
	elseif typeof(test_result) !== typeof(actual_result)
		keep_working(md"The type of the output is not correct. Did you return the required outputs?")
	else
		if !(all(fieldnames(CTMRGEnvironment)) do f 
			getproperty(test_result, f) ≈ getproperty(actual_result, f)
		end)
			almost(md"The returned value is incorrect. Did you take the norm of the difference?")
		else
			test_result2 = singular_value_distance([0.9, 0.8], [0.4, 0.2, 0.1])
			actual_result2 = singular_value_distance_solution([0.9, 0.8], [0.4, 0.2, 0.1])
			if !(test_result2 ≈ actual_result2)
				almost(md"The function does not properly handle inputs of different lengths")
			else
				correct()
			end
		end
	end
end

# ╔═╡ 618d268f-869b-434b-a3d7-380d2f51ef25
let
	test_result = ctmrg_iteration(test_environment, test_doublepeps)
	actual_result = ctmrg_iteration_solution(test_environment, test_doublepeps)
	if ismissing(test_result)
		still_missing()
	elseif typeof(test_result) !== typeof(actual_result)
		keep_working(md"The type of the output is not correct. Did you return the required outputs?")
	else
		if !(all(fieldnames(CTMRGEnvironment)) do f 
			getproperty(test_result, f) ≈ getproperty(actual_result, f)
		end)
			almost(md"The returned value is incorrect. Did you take the norm of the difference?")
		else
			test_result2 = singular_value_distance([0.9, 0.8], [0.4, 0.2, 0.1])
			actual_result2 = singular_value_distance_solution([0.9, 0.8], [0.4, 0.2, 0.1])
			if !(test_result2 ≈ actual_result2)
				almost(md"The function does not properly handle inputs of different lengths")
			else
				correct()
			end
		end
	end
end

# ╔═╡ 8afc3b09-84fe-4966-823d-464805edc66f
let
	test_result = singular_value_distance([0.9, 0.8], [0.4, 0.2])
	actual_result = singular_value_distance_solution([0.9, 0.8], [0.4, 0.2])
	if ismissing(test_result)
		still_missing()
	elseif typeof(test_result) !== typeof(actual_result)
		keep_working(md"The type of the output is not correct. Did you return the required outputs?")
	else
		if !(test_result ≈ actual_result)
			almost(md"The returned value is incorrect. Did you take the norm of the difference?")
		else
			test_result2 = singular_value_distance([0.9, 0.8], [0.4, 0.2, 0.1])
			actual_result2 = singular_value_distance_solution([0.9, 0.8], [0.4, 0.2, 0.1])
			if !(test_result2 ≈ actual_result2)
				almost(md"The function does not properly handle inputs of different lengths")
			else
				correct()
			end
		end
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
OptimKit = "77e91f04-9b3b-57a6-a776-40b61faaebe0"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
TensorKit = "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"
TensorKitTensors = "41b62e7d-e9d1-4e23-942c-79a97adf954b"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
OptimKit = "~0.4.2"
PlutoTeachingTools = "~0.4.6"
PlutoUI = "~0.7.77"
TensorKit = "~0.16.0"
TensorKitTensors = "~0.2.3"
Zygote = "~0.7.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "22f32bb2c8b5712cd4e6dfad07a7227606eeb5db"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

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

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

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

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3b704353e517a957323bd3ac70fa7b669b5f48d4"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.6"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkSplitters]]
git-tree-sha1 = "63a3903063d035260f0f6eab00f517471c5dc784"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.1.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

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

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "5bfcd42851cf2f1b303f51525a54dc5e98d408a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.15.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "b2977f86ed76484de6f29d5b36f2fa686f085487"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.1"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.HalfIntegers]]
git-tree-sha1 = "9c3149243abb5bc0bad0431d6c4fcac0f4443c7c"
uuid = "f0d1745a-41c9-11e9-1dd9-e5d34d218721"
version = "1.6.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LRUCache]]
git-tree-sha1 = "5519b95a490ff5fe629c4a7aa3b3dfc9160498b3"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.2"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MatrixAlgebraKit]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "df89e0216ca068a633b65449fd750dba7905187d"
uuid = "6c742aac-3347-4629-af66-fc926824e5e4"
version = "0.6.2"

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

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "ScopedValues", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "5ece5a3bbfe756517da7b9f1969a66f92fe62ad4"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.8.4"
weakdeps = ["Markdown"]

    [deps.OhMyThreads.extensions]
    MarkdownExt = "Markdown"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OptimKit]]
deps = ["LinearAlgebra", "Printf", "ScopedValues", "VectorInterface"]
git-tree-sha1 = "5c92e3ab480969e80996587da5395f00224b9fdf"
uuid = "77e91f04-9b3b-57a6-a776-40b61faaebe0"
version = "0.4.2"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "dacc8be63916b078b592806acd13bb5e5137d7e9"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "6ed167db158c7c1031abf3bd67f8e689c8bdf2b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.77"

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

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

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

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

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

[[deps.TensorKitTensors]]
deps = ["LinearAlgebra", "TensorKit"]
git-tree-sha1 = "053ec35ef46c8865a712a0cca51b5559cfb57c33"
uuid = "41b62e7d-e9d1-4e23-942c-79a97adf954b"
version = "0.2.3"

[[deps.TensorOperations]]
deps = ["LRUCache", "LinearAlgebra", "PackageExtensionCompat", "PrecompileTools", "Preferences", "PtrArrays", "Strided", "StridedViews", "TupleTools", "VectorInterface"]
git-tree-sha1 = "b1e835ba4d9c073169a63a0ff459577088aea4e6"
uuid = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"
version = "5.4.0"

    [deps.TensorOperations.extensions]
    TensorOperationsBumperExt = "Bumper"
    TensorOperationsChainRulesCoreExt = "ChainRulesCore"
    TensorOperationscuTENSORExt = ["cuTENSOR", "CUDA"]

    [deps.TensorOperations.weakdeps]
    Bumper = "8ce10254-0962-460f-a3d8-1f77fea1446e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a29cbf3968d36022198bcc6f23fdfd70f7caf737"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.10"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─7d26a11c-f98c-473a-9d11-9c564deb1272
# ╟─aeeae089-8fd3-4f85-b0bb-8773f9696073
# ╟─90e7d0b6-79f6-4587-85f9-5f1b68a79fdc
# ╟─9b2b4b2e-eb14-11f0-b4d4-79004a9b7521
# ╠═043300a2-2333-4552-a9c1-d95dead19e51
# ╠═470aa499-b5b8-440b-bd48-095aac2377cf
# ╟─8b70a006-1f58-4181-ae67-f5dbb8a733a7
# ╟─715a210a-7f7b-438a-8b08-b1494dbf99db
# ╠═02cf1c6d-c588-4f49-8de0-4693ed33d21d
# ╟─e995f859-91df-4a8b-9826-11a31c9fa667
# ╠═e37310a8-1e24-4766-bb62-55d0154815bb
# ╟─00b88ea4-3675-41a4-905e-a8103e23e4d6
# ╠═e6325d88-0ae6-47e8-802f-293db7b7f560
# ╟─391f207c-79c0-419e-adf4-b66de67e6d57
# ╟─6af70e05-d625-40ee-ae29-d8b24379eb30
# ╟─b74fcb1a-6252-4350-b274-7b05b35dc9a8
# ╠═cbfc4daf-c79c-4e2a-a70a-f5208cf31975
# ╟─439133a3-10c6-4da6-a671-38d2453d086b
# ╟─d05ac00c-1e69-4952-9629-80468f02096d
# ╠═30e688b4-293f-40af-a839-31fabec981cc
# ╟─d73e61f1-622e-49b2-97bb-dec56807d117
# ╟─f9a33166-f457-4cf9-9409-9917c23d51fe
# ╟─aee1fb88-0631-483f-a7fd-7ecf8c66f06b
# ╟─9577bf9b-1ce2-4497-8de4-5b7ce5e86010
# ╟─32944d9c-c5ad-48eb-a9b0-69ed0df03acf
# ╠═af4bac3d-02cc-48d5-ae3a-df31c7f3e82a
# ╟─b245f91b-df80-4a1b-a6f9-d1b3ceb018da
# ╠═a4abe1b6-f97a-4d81-abc3-4e930dff7632
# ╟─c5838a61-56f4-49f6-b7d7-e1d738aa51af
# ╟─56715c2b-8283-4a83-af08-fc72953b47e7
# ╠═53dc3a3d-db99-4ad9-8889-3afcd5864d84
# ╟─85dc0714-2f66-4bac-975e-a8af43fb4cbc
# ╟─0600518e-aef2-4f9c-ad0f-0f082c58f391
# ╠═faa5861e-bd36-4503-89bf-ed56a190250f
# ╟─394f7270-12f5-4ec0-b7b1-e8e5cd8c6f4e
# ╠═f176dc62-b1f5-447c-958f-b08a6ff2efd7
# ╠═ea38966c-c34d-4e5a-ad74-5cdc5d9c8c77
# ╟─df08983e-dab0-4a00-bf0c-4eb91f42f1fa
# ╟─20f29190-2e0d-42e3-af82-11761478d29d
# ╟─ab8cc3a0-0fd4-4f11-be1d-40b2a44f53d2
# ╟─d2c17a80-1c4a-49be-b615-ed6eb53b0779
# ╠═a3a87f5e-da1a-45f2-b45b-8df484a30de8
# ╟─9da0bb7d-f918-40a1-871f-a546d7b5f641
# ╟─c0862aa4-715b-412a-97ec-5bdf00a5fbdf
# ╟─f2a65413-604c-45f4-99cc-e4f9abc914be
# ╟─4cb24d7c-e175-4cab-90f1-dc91a1d84ec1
# ╠═5d501f80-2e05-48ed-b62e-52ed723a2e90
# ╠═076f533f-1e61-4172-b118-d0815e8c66cd
# ╠═b2b002e3-5c08-4cc6-89c3-07870bf666d7
# ╟─7a94202f-3487-47c7-923a-61a178257374
# ╟─1f0ba404-8001-4bc3-9132-1ce4a3a2e78d
# ╟─b1c0ca10-e70a-42f9-bb49-ecba7a4c0c4e
# ╠═4bb53195-71b9-42d9-b801-c5bf4a674c14
# ╟─ec6e77bb-0fc5-4730-85af-c37d15b2aeed
# ╠═ff1bbb0b-6ff8-4835-a28f-f6b9ef221faa
# ╟─3983afb7-585b-4e43-aab1-de4ce47d685a
# ╠═effb78fe-894f-4b44-95ea-f3d80134bc75
# ╟─d6a71ccd-5bee-4532-bf29-03348ac7bb8e
# ╟─f5dd768c-97b4-48cc-be0c-ed36ffad371d
# ╠═607be339-8a6a-4a18-bbde-7775a349bbc1
# ╟─941cfb90-087e-48ea-93df-8e72aaaaff8b
# ╠═1c6868d8-49aa-41a5-ac63-a09fa5e60fc4
# ╠═0689c458-4352-4800-a7e2-29b9adfce426
# ╟─0ba5f43a-509f-49e5-ae4d-7d4c995a60fc
# ╠═d7579195-063f-49a5-95b8-8e88dc823321
# ╟─c410317c-9fd6-4c05-8053-544f24534a0f
# ╠═e4bda769-1aaf-4bcc-a2ca-600adb988639
# ╟─b986149d-055a-4588-a08f-e2a86e803a38
# ╠═f95272cc-1b29-479f-a1ec-3128bf393204
# ╟─2506dde2-acac-41c9-935e-023b87223a8f
# ╠═0d6f69d6-7ed0-487a-852b-241d2580e1f3
# ╟─05e524c3-0369-4bea-8aea-eec522953d5f
# ╠═f7bd71f9-f785-452a-a606-84d0f823f09d
# ╠═618d268f-869b-434b-a3d7-380d2f51ef25
# ╟─3d4a07e4-f5c1-42d7-bc4d-a15a2fb36b66
# ╠═2481bcb0-3c48-4d3e-97d4-5f00efd8ef2f
# ╠═8c32a473-4f08-4955-bdea-25cef4490de0
# ╟─8afc3b09-84fe-4966-823d-464805edc66f
# ╟─df668aea-2156-40f4-8dbd-2a442e863210
# ╟─67503430-d5d1-4ee6-847f-4b7dbecdf7ed
# ╟─80907dcf-e9b6-47e8-89e9-f9e69a23ee63
# ╟─f6acc633-d9ab-4cf5-b475-f608d0df13b2
# ╟─d007a99a-fb6a-4b76-9edc-949fa2787b8d
# ╟─d97aae83-b513-4332-bc01-17fdd38b1381
# ╠═f657e893-c238-419a-941a-f0ed9374e838
# ╠═1c8c0edc-4ea6-470a-8b6c-ad4e9e2cb230
# ╠═085669a2-0c44-4df2-93ce-475be4e45579
# ╠═06bb1f23-a4cb-4b10-9d4e-b8f59c2171cc
# ╠═7eb8edae-9d1b-4311-af50-6a258840f8eb
# ╠═cb153168-3a38-42b1-a2e5-4b5cd1062e49
# ╠═d6e0fb57-f686-4400-aa92-748288e0591d
# ╠═e58ed93c-4b37-44c4-8269-bb8cb039b4fa
# ╠═209aa00c-0281-4e8c-b32c-d77888e8a5ae
# ╟─b333c5f9-ba3d-429c-9403-37d1f9fb7360
# ╟─dececc2a-5b73-4ad8-8056-e5c0a97a230f
# ╠═f897eafd-d376-4ed9-811b-76290bf11280
# ╠═14f49304-2286-48aa-b1b7-07a2e6f03cfe
# ╠═f560e045-433f-4562-a611-119ce410342f
# ╠═b9ae7480-9341-4293-b615-d4b353ca39ed
# ╠═71526bec-030b-476c-a264-db69f858aeb7
# ╠═99fad623-4b57-4aad-bc3a-66230fb81034
# ╠═49d2008e-8030-4c5f-8f98-cb2c42fd5d7d
# ╠═ada2cab6-fc02-4b05-a7d5-edbbae746dde
# ╠═48270af5-c36a-499d-b0c1-f688820e7b84
# ╠═3411519b-ac28-4983-b477-026c8d015b97
# ╠═80afa573-0cc0-4c7e-98d6-f58bdf9205d9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
