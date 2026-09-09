//! Native, deterministic nonlinear least-squares fitting for spectroscopy.
//!
//! The crate is independent from Spectrix's UI and from Python. Its high-level
//! spectrum API reproduces Spectrix's Gaussian peak preprocessing while the
//! lower-level [`Model`] and [`FitProblem`] APIs support custom/composite models.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod model;
mod parameter;
mod solver;
mod spectrum;

pub use error::FitError;
pub use model::{
    ComponentCurve, CompositeModel, ConstantModel, ExponentialModel, GaussianModel, LinearModel,
    Model, ModelComponent, PowerLawModel, QuadraticModel,
};
pub use parameter::{
    Bound, Bounds, DerivedParameter, ParameterBinding, ParameterDefinition, ParameterEstimate,
    ParameterKind, ParameterValues,
};
pub use solver::{
    ConfidenceBand, Covariance, CovarianceStatus, FitDiagnostics, FitOptions, FitProblem,
    FitResult, FitStatistics, ObjectiveKind, SolverProfile, TerminationStatus, fit,
};
pub use spectrum::{
    BackgroundCoupling, BackgroundFitRequest, BackgroundKind, BackgroundSeed, FitQualityIssue,
    FitQualityStatus, ManualPeakBounds, ManualPeakEstimate, ManualPeakSeed, ManualSeedEstimate,
    ManualSeedEstimateRequest, PeakFitRequest, SigmaBounds, SpectrumFitResult,
    estimate_manual_peak_seeds, evaluate_manual_peak, fit_background, fit_peaks,
};
