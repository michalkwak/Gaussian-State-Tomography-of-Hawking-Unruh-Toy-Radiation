# Horizon Radiation Tomography
 
Simulating and reconstructing simplified Hawking/Unruh radiation through
Gaussian quantum-state tomography.
 
## Overview
 
Horizons - black hole event horizons - are predicted to generate thermal particle radiation (Hawking and
Unruh radiation) from correlations between an observable mode and a
partner mode hidden behind the horizon. This project builds a toy model of
that process and asks a concrete, quantitative question - given only
finite, noisy measurements of the accessible mode, how much can be
recovered about the underlying quantum state?
 
The horizon is modeled as a two-mode squeezed Gaussian state: one mode
represents radiation an outside observer can measure, the other an
inaccessible partner mode beyond the horizon. Tracing out the hidden mode
leaves the observer with a thermal mixed state - this is the toy-model
analogue of Hawking radiation appearing thermal despite arising from a
globally pure, entangled state.
 
The pipeline:
1. Generate the exact covariance matrix of the accessible mode from a
   squeezing parameter `r`.
2. Simulate realistic, noisy quadrature measurements of that mode under
   different detection schemes.
3. Reconstruct the covariance matrix from finite samples.
4. Estimate physical quantities - squeezing parameter, particle number,
   entanglement entropy - from the reconstruction, and quantify the
   recovery error against ground truth.
5. Sweep over squeezing strength, sample size, noise level, and detection
   scheme to characterize how estimation accuracy degrades under realistic
   experimental constraints.
   
## Physics background
 
A two-mode squeezed vacuum state has covariance matrix (vacuum
normalized to identity):
 
```
V(r) = [ cosh(2r)·I₂      sinh(2r)·Z  ]
       [ sinh(2r)·Z       cosh(2r)·I₂ ]
```
 
Tracing out the hidden mode gives the reduced state of the accessible
mode, `V_A = cosh(2r)·I₂` - a thermal state whose properties are fully
determined by its symplectic eigenvalue `ν = sqrt(det(V_A))`:
 
- **Particle number:** `N = (Tr(V_A) − 2) / 4`
- **Entanglement entropy:** `S = g(ν)`, the standard thermal-state entropyfunction of a Gaussian mode

Three detection schemes are modeled, each with a genuinely different
information/noise trade-off:
 
- **Joint** - both quadratures measured every shot (idealized baseline).
- **Homodyne** - only one quadrature measured per shot, the other
  unobserved; recovering the full covariance requires handling missing
  data rather than a plain sample covariance.
- **Heterodyne** - both quadratures measured every shot, but with an
  extra unit of vacuum noise added - a fundamental measurement penalty,
  not a technical limitation.
