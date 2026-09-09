# spectrix-fitting

`spectrix-fitting` is a safe, deterministic Rust crate for nonlinear least-squares and Poisson-deviance fitting of spectra. It has no Python or UI dependency and does not expose its internal linear-algebra types. Version 0.1 provides:

- unit-normalized Gaussian peaks (`amplitude` is the integral), including `height`, `fwhm`, and Spectrix `area = amplitude / bin_width`;
- none/constant, linear, quadratic, exponential, and power-law backgrounds;
- fixed, bounded, shared, and derived parameters;
- likelihood covariance for Poisson fits and reduced-chi-square-scaled covariance for least squares;
- normal-based Poisson confidence bands and Student-t-scaled least-squares bands;
- a MINPACK-compatible `Lmfit134` solver profile; and
- spectrum preprocessing compatible with Spectrix marker workflows.

The crate forbids unsafe code and uses checked allocation limits. Singular covariance is returned as unavailable; it is never synthesized.

## Peak fitting

```rust
use spectrix_fitting::{
    fit_peaks, BackgroundCoupling, BackgroundKind, FitOptions, ManualPeakSeed,
    ObjectiveKind, PeakFitRequest,
};

let x = (0..101).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
let y = x.iter().map(|x| {
    2.0 + 80.0 / ((2.0 * std::f64::consts::PI).sqrt() * 0.3)
        * (-0.5 * ((*x - 5.0) / 0.3).powi(2)).exp()
}).collect::<Vec<_>>();

let request = PeakFitRequest {
    x,
    y,
    bin_width: 0.1,
    region: [2.0, 8.0],
    peak_seeds: vec![ManualPeakSeed { center: 5.0, sigma: 0.3, amplitude: 80.0 }],
    peak_bounds: None,
    background_markers: vec![(2.0, 3.0), (7.0, 8.0)],
    background: BackgroundKind::Constant,
    background_seed: None,
    background_coupling: BackgroundCoupling::PrefitJoint,
    equal_sigma: true,
    free_centers: true,
    sigma_bounds: None,
};

let mut options = FitOptions::robust();
options.objective = ObjectiveKind::PoissonDeviance;
let result = fit_peaks(&request, &options)?;
assert!(result.fit.termination.success);
assert!(result.fit.covariance.is_some());
assert!(result.fit.confidence_band.is_some());
# Ok::<(), spectrix_fitting::FitError>(())
```

Use `FitOptions::robust()` for new fits. The application uses this profile automatically. It moves boundary starting values slightly inward without changing user bounds, checks physical projected gradients, and polishes the complete solution. Failed stationarity or uncertainty checks trigger three deterministic width alternatives for peak fits. All optimizer attempts share `evaluation_patience * (nvarys + 1)` residual evaluations; the lowest finite objective is retained, regardless of covariance availability. `FitOptions::default()` retains the `Lmfit134` compatibility profile and its single-solve behavior.

Choose the background family explicitly. With the robust profile, empty background windows request a peak-resistant asymmetric least-squares initialization over the region (at most ten passes, weights 0.05 above the estimate and 0.95 below). These weights are only for initialization: the final joint fit uses the selected objective. Supplied windows initialize the background and, for `BackgroundCoupling::PrefitJoint`, remain in the joint fit as a union with the region. Each original bin is counted once, including external windows; the full peak-plus-background equation is evaluated there. `observation_x` identifies the residual grid, while `evaluation_x` remains the display grid. The compatibility profile still requires explicit windows for non-`None` backgrounds.

The application uses `BackgroundCoupling::PrefitFrozen` whenever background windows are supplied: it fits the selected background to those bins and holds the result fixed during peak fitting. The library still offers `PrefitJoint` explicitly. Use `PrefitFrozen` with fixed seed values to hold an existing background result (the application's **Lock manual background** workflow). Peak uncertainties then condition on the fixed background and exclude its uncertainty. Optional `ManualPeakBounds` constrain position, sigma, and net bin height one-for-one with the required seeds. The fitter varies net height directly, then reports integrated amplitude and area as derived parameters; recovery never changes peak count, fixed values, shared-width constraints, or user bounds.

Robust covariance comes from a column-scaled SVD of the physical information Jacobian. Least-squares covariance uses reduced-chi-square scaling; Poisson covariance uses expected information. Unconverged, rank-deficient, and boundary-limited solutions retain fitted values but withhold symmetric uncertainties. `FitResult::diagnostics` reports attempts, stationarity, rank when available, affected parameters, and budget exhaustion. New diagnostics and observation coordinates default safely when reading older saved results.

Histogram Gaussians are integrated across bin edges while retaining Spectrix's existing amplitude and area conventions.

## Custom and composite models

Implement the object-safe `Model` trait for a custom equation. Parameters use names rather than public nalgebra types, and an analytic Jacobian is optional:

```rust
use spectrix_fitting::{
    FitError, Model, ParameterDefinition, ParameterValues,
};

struct Offset;

impl Model for Offset {
    fn name(&self) -> &str { "offset" }

    fn parameter_definitions(&self) -> Vec<ParameterDefinition> {
        vec![ParameterDefinition::varying("offset", 0.0)]
    }

    fn evaluate(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        if x.len() != output.len() {
            return Err(FitError::LengthMismatch { x: x.len(), y: output.len() });
        }
        output.fill(parameters.require("offset")?);
        Ok(())
    }
}
```

Built-ins can be combined with `CompositeModel` and `ModelComponent`. Prefix each component's parameter names (for example `g0_` and `g1_`) so the namespace remains unique. `ParameterDefinition::equal_to` creates shared bindings such as equal sigma.

## Compatibility testing

The committed oracles in `tests/parity` are generated with lmfit 1.3.4, NumPy 2.5.2, and SciPy 1.18.1. They cover every V1 background equation and a high-level matrix of one, three, and overlapping peaks; equal and independent sigma; fixed and free centers; sigma constraints; marker fallback/filtering; reversed regions; frozen and joint coupling; and bounded background seeds. Regenerate them only in that pinned environment, then run:

```text
cargo test -p spectrix-fitting --all-features
```

Compatibility thresholds are `rtol=1e-8, atol=1e-10` for fitted values, curves, residuals, and statistics, and `rtol=1e-6, atol=1e-9` for covariance, correlations, errors, and bands.

## Performance gate

Run `benches/compare.ps1` from the workspace root with the pinned parity environment. The warmed release-mode matrix covers 1, 3, and 8 fixed-center peaks over 512, 2048, and 8192 bins, including preprocessing, solve, covariance, component curves, and complete total/component band payloads. The committed [`PERFORMANCE.md`](PERFORMANCE.md) records the latest passing run.

## License

Licensed under either Apache-2.0 or MIT, at your option.
