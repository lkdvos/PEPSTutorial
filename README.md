# PEPS & CTMRG Workshop — Setup Instructions

This repository contains interactive **Pluto notebooks** for a hands-on tutorial on PEPS algorithms, CTMRG, and variational optimization using **TensorKit.jl** and the Julia ecosystem.

This README explains how to:

1. Install **Julia (LTS)** using **Juliaup**,
2. Verify the installation,
3. Launch the Pluto notebooks.

---

## 1. Install Julia (LTS) using Juliaup

Juliaup is the official Julia version manager and the **recommended** way to install Julia across platforms.

### Linux and macOS

Open a terminal and run:

```bash
curl -fsSL https://install.julialang.org | sh
```

Restart your terminal after installation.

### Windows

Install Juliaup from the Microsoft Store:

* Search for **“Julia”** in the Microsoft Store, or
* Visit: [https://www.microsoft.com/store/apps/9NJNWW8PVKMN](https://www.microsoft.com/store/apps/9NJNWW8PVKMN)

---

## 2. Install and select the Julia LTS version

Once Juliaup is installed, install the **Long-Term Support (LTS)** release:

```bash
juliaup add lts
```

Set it as the default Julia version:

```bash
juliaup default lts
```

Verify that Julia is available and that the LTS version is active:

```bash
julia --version
```

You should see output indicating a Julia **LTS** release (e.g. `1.10.x LTS`).

---

## 3. Clone this repository

Clone the workshop repository and enter the directory:

```bash
git clone https://github.com/lkdvos/PEPSTutorial
cd PEPSTutorial
```

---

## 4. Start Julia and launch Pluto

From the repository root, start Julia:

```bash
julia
```

Inside the Julia REPL, activate the project environment and install dependencies:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then launch Pluto:

```julia
using Pluto
Pluto.run()
```

This will open Pluto in your default web browser (typically at `http://localhost:1234`).

---

## 5. Open the notebooks

In the Pluto web interface:

1. Click **“Open a notebook”**,
2. Navigate to the notebook directory and open the notebook `notebooks/CTMRG.jl`

Pluto notebooks are **reactive**:

* Cells automatically re-run when dependencies change,
* You do not need to manually “run all cells”.

---

## 6. Troubleshooting

### Pluto does not open automatically

Pluto prints a local URL in the terminal. Copy and paste it into your browser manually.

### Package installation fails

Ensure:

* You are using the **LTS** Julia version,
* You have an active internet connection.


### Performance is very slow

* Start with smaller bond dimensions (`D`, `χ`) as suggested in the notebooks.

### Various other issues

Please feel free to ask or open an issue!

