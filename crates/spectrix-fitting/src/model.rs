use std::collections::BTreeSet;

use statrs::function::erf::erf;

use crate::{Bounds, DerivedParameter, FitError, ParameterDefinition, ParameterValues};

const SQRT_TWO_PI: f64 = 2.506_628_274_631_000_2;
// lmfit 1.3.4's GaussianModel expression deliberately uses this rounded factor.
const FWHM_FACTOR: f64 = 2.354_82;
const HEIGHT_FACTOR: f64 = 0.398_942_3;

/// A UI-independent model suitable for nonlinear least-squares fitting.
pub trait Model: Send + Sync {
    /// Human-readable model name.
    fn name(&self) -> &str;

    /// Defines every parameter used by this model.
    fn parameter_definitions(&self) -> Vec<ParameterDefinition>;

    /// Evaluates the model into `output`, which must match `x.len()`.
    fn evaluate(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        output: &mut [f64],
    ) -> Result<(), FitError>;

    /// Optionally evaluates a row-major Jacobian for the requested parameter names.
    ///
    /// Returning `Ok(false)` asks the fitting engine to use compatibility-profile
    /// finite differences. The output has `x.len() * parameter_names.len()` entries.
    fn analytic_jacobian(
        &self,
        _x: &[f64],
        _parameters: &ParameterValues,
        _parameter_names: &[String],
        _output: &mut [f64],
    ) -> Result<bool, FitError> {
        Ok(false)
    }

    /// Analytic derivatives used by the robust profile. Returning false enables
    /// scaled physical-coordinate finite differences without changing legacy fits.
    fn robust_jacobian(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        output: &mut [f64],
    ) -> Result<bool, FitError> {
        self.analytic_jacobian(x, parameters, parameter_names, output)
    }

    /// Optionally evaluates row-major Jacobians for every value returned by
    /// [`Model::components`].
    ///
    /// `output` has one vector per component, and each vector has
    /// `x.len() * parameter_names.len()` entries. The default supports a
    /// single-component model by forwarding to [`Model::analytic_jacobian`].
    fn analytic_component_jacobians(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        output: &mut [Vec<f64>],
    ) -> Result<bool, FitError> {
        if output.len() != 1 {
            return Ok(false);
        }
        self.analytic_jacobian(x, parameters, parameter_names, &mut output[0])
    }

    /// Analytic component derivatives for robust uncertainty propagation.
    fn robust_component_jacobians(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        output: &mut [Vec<f64>],
    ) -> Result<bool, FitError> {
        if output.len() != 1 {
            return Ok(false);
        }
        self.robust_jacobian(x, parameters, parameter_names, &mut output[0])
    }

    /// Optionally evaluates lmfit-compatible central-difference component Jacobians.
    ///
    /// `steps` contains one external-parameter step per `parameter_names` entry.
    /// Implementations may skip unrelated parameters, but must reproduce evaluating
    /// the component at `value + step` and `value - step` for related parameters.
    fn compatibility_component_jacobians(
        &self,
        _x: &[f64],
        _parameters: &ParameterValues,
        _parameter_names: &[String],
        _steps: &[f64],
        _output: &mut [Vec<f64>],
    ) -> Result<bool, FitError> {
        Ok(false)
    }

    /// Computes derived parameters and their gradients.
    fn derived_parameters(
        &self,
        _parameters: &ParameterValues,
    ) -> Result<Vec<DerivedParameter>, FitError> {
        Ok(Vec::new())
    }

    /// Evaluates named component curves.
    fn components(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
    ) -> Result<Vec<ComponentCurve>, FitError> {
        let mut values = checked_zeros(x.len())?;
        self.evaluate(x, parameters, &mut values)?;
        Ok(vec![ComponentCurve {
            name: self.name().to_owned(),
            values,
        }])
    }
}

/// A named model component evaluated on the fit grid.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ComponentCurve {
    /// Component name or prefix.
    pub name: String,
    /// Evaluated component values.
    pub values: Vec<f64>,
}

/// An owned component of a [`CompositeModel`].
pub struct ModelComponent {
    name: String,
    model: Box<dyn Model>,
}

impl ModelComponent {
    /// Creates a named component from any model.
    #[must_use]
    pub fn new(name: impl Into<String>, model: Box<dyn Model>) -> Self {
        Self {
            name: name.into(),
            model,
        }
    }
}

/// A sum of prefixed model components.
#[derive(Default)]
pub struct CompositeModel {
    components: Vec<ModelComponent>,
}

impl CompositeModel {
    /// Creates a model from named components.
    pub fn new(components: Vec<ModelComponent>) -> Result<Self, FitError> {
        let model = Self { components };
        let mut names = BTreeSet::new();
        for definition in model.parameter_definitions() {
            if !names.insert(definition.name.clone()) {
                return Err(FitError::InvalidParameter {
                    parameter: definition.name,
                });
            }
        }
        Ok(model)
    }

    /// Appends a named component, rejecting duplicate parameter names.
    pub fn push(&mut self, component: ModelComponent) -> Result<(), FitError> {
        self.components.push(component);
        let mut names = BTreeSet::new();
        if let Some(duplicate) = self
            .parameter_definitions()
            .into_iter()
            .find(|definition| !names.insert(definition.name.clone()))
        {
            self.components.pop();
            return Err(FitError::InvalidParameter {
                parameter: duplicate.name,
            });
        }
        Ok(())
    }
}

impl Model for CompositeModel {
    fn name(&self) -> &'static str {
        "composite"
    }

    fn parameter_definitions(&self) -> Vec<ParameterDefinition> {
        self.components
            .iter()
            .flat_map(|component| component.model.parameter_definitions())
            .collect()
    }

    fn evaluate(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        check_output(x, output)?;
        output.fill(0.0);
        let mut temporary = checked_zeros(x.len())?;
        for component in &self.components {
            component.model.evaluate(x, parameters, &mut temporary)?;
            for (sum, value) in output.iter_mut().zip(&temporary) {
                *sum += value;
            }
        }
        ensure_finite(output, self.name())
    }

    fn analytic_jacobian(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        output: &mut [f64],
    ) -> Result<bool, FitError> {
        let expected = x.len().saturating_mul(parameter_names.len());
        if output.len() != expected {
            return Err(FitError::LengthMismatch {
                x: expected,
                y: output.len(),
            });
        }
        output.fill(0.0);
        let mut temporary = checked_zeros(expected)?;
        for component in &self.components {
            temporary.fill(0.0);
            if !component
                .model
                .analytic_jacobian(x, parameters, parameter_names, &mut temporary)?
            {
                return Ok(false);
            }
            for (sum, value) in output.iter_mut().zip(&temporary) {
                *sum += value;
            }
        }
        Ok(true)
    }

    fn robust_jacobian(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        output: &mut [f64],
    ) -> Result<bool, FitError> {
        let expected = x.len().saturating_mul(parameter_names.len());
        if output.len() != expected {
            return Err(FitError::LengthMismatch {
                x: expected,
                y: output.len(),
            });
        }
        output.fill(0.0);
        let mut temporary = checked_zeros(expected)?;
        for component in &self.components {
            temporary.fill(0.0);
            if !component
                .model
                .robust_jacobian(x, parameters, parameter_names, &mut temporary)?
            {
                return Ok(false);
            }
            for (sum, value) in output.iter_mut().zip(&temporary) {
                *sum += value;
            }
        }
        Ok(true)
    }

    fn analytic_component_jacobians(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        output: &mut [Vec<f64>],
    ) -> Result<bool, FitError> {
        if output.len() != self.components.len() {
            return Ok(false);
        }
        for (component, component_output) in self.components.iter().zip(output) {
            component_output.fill(0.0);
            if !component.model.analytic_jacobian(
                x,
                parameters,
                parameter_names,
                component_output,
            )? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn robust_component_jacobians(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        output: &mut [Vec<f64>],
    ) -> Result<bool, FitError> {
        if output.len() != self.components.len() {
            return Ok(false);
        }
        for (component, component_output) in self.components.iter().zip(output) {
            component_output.fill(0.0);
            if !component
                .model
                .robust_jacobian(x, parameters, parameter_names, component_output)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn compatibility_component_jacobians(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        steps: &[f64],
        output: &mut [Vec<f64>],
    ) -> Result<bool, FitError> {
        if output.len() != self.components.len() {
            return Ok(false);
        }
        for (component, component_output) in self.components.iter().zip(output) {
            component_output.fill(0.0);
            if !component.model.compatibility_component_jacobians(
                x,
                parameters,
                parameter_names,
                steps,
                std::slice::from_mut(component_output),
            )? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn derived_parameters(
        &self,
        parameters: &ParameterValues,
    ) -> Result<Vec<DerivedParameter>, FitError> {
        self.components
            .iter()
            .try_fold(Vec::new(), |mut all, component| {
                all.extend(component.model.derived_parameters(parameters)?);
                Ok(all)
            })
    }

    fn components(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
    ) -> Result<Vec<ComponentCurve>, FitError> {
        self.components
            .iter()
            .map(|component| {
                let mut values = checked_zeros(x.len())?;
                component.model.evaluate(x, parameters, &mut values)?;
                Ok(ComponentCurve {
                    name: component.name.clone(),
                    values,
                })
            })
            .collect()
    }
}

/// A unit-normalized Gaussian whose amplitude is the integral.
#[derive(Debug, Clone)]
pub struct GaussianModel {
    prefix: String,
    amplitude: ParameterDefinition,
    center: ParameterDefinition,
    sigma: ParameterDefinition,
    bin_width: Option<f64>,
    integrate_bins: bool,
}

impl GaussianModel {
    /// Creates a Gaussian with parameters `{prefix}amplitude`, `{prefix}center`, and `{prefix}sigma`.
    #[must_use]
    pub fn new(prefix: impl Into<String>, amplitude: f64, center: f64, sigma: f64) -> Self {
        let prefix = prefix.into();
        Self {
            amplitude: ParameterDefinition::varying(format!("{prefix}amplitude"), amplitude),
            center: ParameterDefinition::varying(format!("{prefix}center"), center),
            sigma: ParameterDefinition::varying(format!("{prefix}sigma"), sigma)
                .with_bounds(Bounds::lower_bounded(0.0)),
            prefix,
            bin_width: None,
            integrate_bins: false,
        }
    }

    /// Sets the parameter definitions used by this Gaussian.
    #[must_use]
    pub fn with_parameters(
        mut self,
        amplitude: ParameterDefinition,
        center: ParameterDefinition,
        sigma: ParameterDefinition,
    ) -> Self {
        self.amplitude = amplitude;
        self.center = center;
        self.sigma = sigma;
        self
    }

    /// Enables the Spectrix area derived parameter (`amplitude / bin_width`).
    #[must_use]
    pub const fn with_bin_width(mut self, bin_width: f64) -> Self {
        self.bin_width = Some(bin_width);
        self
    }

    /// Evaluates expected bin counts by integrating the Gaussian over each bin.
    ///
    /// The public amplitude convention is unchanged: Spectrix area remains
    /// `amplitude / bin_width`.
    #[must_use]
    pub const fn with_bin_integration(mut self, bin_width: f64) -> Self {
        self.bin_width = Some(bin_width);
        self.integrate_bins = true;
        self
    }
}

impl Model for GaussianModel {
    fn name(&self) -> &str {
        &self.prefix
    }

    fn parameter_definitions(&self) -> Vec<ParameterDefinition> {
        vec![
            self.amplitude.clone(),
            self.center.clone(),
            self.sigma.clone(),
        ]
    }

    fn evaluate(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        output: &mut [f64],
    ) -> Result<(), FitError> {
        check_output(x, output)?;
        let amplitude = parameters.require(&self.amplitude.name)?;
        let center = parameters.require(&self.center.name)?;
        let sigma = parameters.require(&self.sigma.name)?;
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(FitError::Domain {
                model: self.prefix.clone(),
                message: "sigma must be positive and finite".to_owned(),
            });
        }
        if self.integrate_bins {
            let bin_width = self.bin_width.ok_or_else(|| FitError::Domain {
                model: self.prefix.clone(),
                message: "bin integration requires a positive bin width".to_owned(),
            })?;
            if !bin_width.is_finite() || bin_width <= 0.0 {
                return Err(FitError::Domain {
                    model: self.prefix.clone(),
                    message: "bin integration requires a positive bin width".to_owned(),
                });
            }
            let denominator = sigma * std::f64::consts::SQRT_2;
            for (value, independent) in output.iter_mut().zip(x) {
                let lower = (*independent - 0.5 * bin_width - center) / denominator;
                let upper = (*independent + 0.5 * bin_width - center) / denominator;
                *value = amplitude / bin_width * 0.5 * (erf(upper) - erf(lower));
            }
        } else {
            let scale = amplitude / (SQRT_TWO_PI * sigma);
            for (value, independent) in output.iter_mut().zip(x) {
                let standardized = (*independent - center) / sigma;
                *value = scale * (-0.5 * standardized * standardized).exp();
            }
        }
        ensure_finite(output, &self.prefix)
    }

    fn analytic_jacobian(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        output: &mut [f64],
    ) -> Result<bool, FitError> {
        if output.len() != x.len().saturating_mul(parameter_names.len()) {
            return Err(FitError::LengthMismatch {
                x: x.len().saturating_mul(parameter_names.len()),
                y: output.len(),
            });
        }
        let amplitude = parameters.require(&self.amplitude.name)?;
        let center = parameters.require(&self.center.name)?;
        let sigma = parameters.require(&self.sigma.name)?;
        if sigma <= 0.0 {
            return Ok(false);
        }
        let amplitude_column = match analytic_column(&self.amplitude, parameter_names) {
            AnalyticColumn::Unrelated => None,
            AnalyticColumn::Column(index) => Some(index),
            AnalyticColumn::UnresolvedBinding => return Ok(false),
        };
        let center_column = match analytic_column(&self.center, parameter_names) {
            AnalyticColumn::Unrelated => None,
            AnalyticColumn::Column(index) => Some(index),
            AnalyticColumn::UnresolvedBinding => return Ok(false),
        };
        let sigma_column = match analytic_column(&self.sigma, parameter_names) {
            AnalyticColumn::Unrelated => None,
            AnalyticColumn::Column(index) => Some(index),
            AnalyticColumn::UnresolvedBinding => return Ok(false),
        };
        output.fill(0.0);
        if self.integrate_bins {
            let Some(bin_width) = self.bin_width.filter(|value| *value > 0.0) else {
                return Ok(false);
            };
            let denominator = sigma * std::f64::consts::SQRT_2;
            for (row, independent) in x.iter().enumerate() {
                let lower = (*independent - 0.5 * bin_width - center) / denominator;
                let upper = (*independent + 0.5 * bin_width - center) / denominator;
                let fraction = 0.5 * (erf(upper) - erf(lower)) / bin_width;
                let offset = row * parameter_names.len();
                if let Some(column) = amplitude_column {
                    output[offset + column] += fraction;
                }
                if let Some(column) = center_column {
                    output[offset + column] += amplitude / bin_width
                        * (lower.mul_add(-lower, 0.0).exp() - upper.mul_add(-upper, 0.0).exp())
                        / (SQRT_TWO_PI * sigma);
                }
                if let Some(column) = sigma_column {
                    output[offset + column] += amplitude / bin_width
                        * (lower * (-lower * lower).exp() - upper * (-upper * upper).exp())
                        / (std::f64::consts::PI.sqrt() * sigma);
                }
            }
            return Ok(true);
        }
        for (row, independent) in x.iter().enumerate() {
            let delta = *independent - center;
            let base = (-0.5 * (delta / sigma).powi(2)).exp() / (SQRT_TWO_PI * sigma);
            let value = amplitude * base;
            let offset = row * parameter_names.len();
            if let Some(column) = amplitude_column {
                output[offset + column] += base;
            }
            if let Some(column) = center_column {
                output[offset + column] += value * delta / sigma.powi(2);
            }
            if let Some(column) = sigma_column {
                output[offset + column] += value * (delta.powi(2) / sigma.powi(3) - 1.0 / sigma);
            }
        }
        Ok(true)
    }

    fn compatibility_component_jacobians(
        &self,
        x: &[f64],
        parameters: &ParameterValues,
        parameter_names: &[String],
        steps: &[f64],
        output: &mut [Vec<f64>],
    ) -> Result<bool, FitError> {
        if output.len() != 1 {
            return Ok(false);
        }
        let columns = parameter_names.len();
        let expected = x.len().saturating_mul(columns);
        if steps.len() != columns || output[0].len() != expected {
            return Err(FitError::LengthMismatch {
                x: expected,
                y: output[0].len(),
            });
        }
        if self.integrate_bins {
            return compatibility_finite_difference(
                self,
                [&self.amplitude, &self.center, &self.sigma],
                x,
                parameters,
                parameter_names,
                steps,
                &mut output[0],
            );
        }
        let amplitude = parameters.require(&self.amplitude.name)?;
        let center = parameters.require(&self.center.name)?;
        let sigma = parameters.require(&self.sigma.name)?;
        let mappings = [
            analytic_column(&self.amplitude, parameter_names),
            analytic_column(&self.center, parameter_names),
            analytic_column(&self.sigma, parameter_names),
        ];
        if mappings.contains(&AnalyticColumn::UnresolvedBinding) {
            return Ok(false);
        }
        let mapped_columns = mappings
            .iter()
            .filter_map(|mapping| match mapping {
                AnalyticColumn::Column(column) => Some(*column),
                AnalyticColumn::Unrelated | AnalyticColumn::UnresolvedBinding => None,
            })
            .collect::<Vec<_>>();
        if mapped_columns
            .iter()
            .enumerate()
            .any(|(index, column)| mapped_columns[index + 1..].contains(column))
        {
            return compatibility_finite_difference(
                self,
                [&self.amplitude, &self.center, &self.sigma],
                x,
                parameters,
                parameter_names,
                steps,
                &mut output[0],
            );
        }
        let column = |mapping: AnalyticColumn| match mapping {
            AnalyticColumn::Column(column) => Some(column),
            AnalyticColumn::Unrelated | AnalyticColumn::UnresolvedBinding => None,
        };
        let amplitude_column = column(mappings[0]);
        let center_column = column(mappings[1]);
        let sigma_column = column(mappings[2]);
        output[0].fill(0.0);
        for (row, independent) in x.iter().enumerate() {
            let delta = *independent - center;
            let standardized = delta / sigma;
            let base = (-0.5 * standardized * standardized).exp() / (SQRT_TWO_PI * sigma);
            let value = amplitude * base;
            let offset = row * columns;
            if let Some(column) = amplitude_column {
                output[0][offset + column] = base;
            }
            if let Some(column) = center_column {
                output[0][offset + column] = value * delta / sigma.powi(2);
            }
            if let Some(column) = sigma_column {
                let step = steps[column];
                let q = standardized * standardized;
                let first = (q - 1.0) / sigma;
                let third = (q.powi(3) - 12.0 * q.powi(2) + 27.0 * q - 6.0) / sigma.powi(3);
                output[0][offset + column] = value * (first + step.powi(2) * third / 6.0);
            }
        }
        Ok(true)
    }

    fn derived_parameters(
        &self,
        parameters: &ParameterValues,
    ) -> Result<Vec<DerivedParameter>, FitError> {
        let amplitude = parameters.require(&self.amplitude.name)?;
        let sigma = parameters.require(&self.sigma.name)?;
        let mut derived = vec![
            DerivedParameter {
                name: format!("{}fwhm", self.prefix),
                value: FWHM_FACTOR * sigma,
                gradient: vec![(self.sigma.name.clone(), FWHM_FACTOR)],
            },
            DerivedParameter {
                name: format!("{}height", self.prefix),
                value: HEIGHT_FACTOR * amplitude / sigma,
                gradient: vec![
                    (self.amplitude.name.clone(), HEIGHT_FACTOR / sigma),
                    (
                        self.sigma.name.clone(),
                        -HEIGHT_FACTOR * amplitude / sigma.powi(2),
                    ),
                ],
            },
        ];
        if let Some(bin_width) = self.bin_width {
            if !bin_width.is_finite() || bin_width <= 0.0 {
                return Err(FitError::Domain {
                    model: self.prefix.clone(),
                    message: "bin width must be positive and finite".to_owned(),
                });
            }
            derived.push(DerivedParameter {
                name: format!("{}area", self.prefix),
                value: amplitude / bin_width,
                gradient: vec![(self.amplitude.name.clone(), 1.0 / bin_width)],
            });
        }
        Ok(derived)
    }
}

macro_rules! simple_model {
    ($type:ident, $doc:literal, $name:literal, [$($field:ident),+], $body:expr, $derivatives:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone)]
        pub struct $type {
            prefix: String,
            $(
                $field: ParameterDefinition,
            )+
        }

        impl $type {
            /// Creates the model from its prefixed initial parameter values.
            #[must_use]
            pub fn new(prefix: impl Into<String>, values: [f64; simple_model!(@count $($field),+)]) -> Self {
                let prefix = prefix.into();
                let mut values = values.into_iter();
                Self {
                    $(
                        $field: ParameterDefinition::varying(
                            format!("{}{}", prefix, stringify!($field)),
                            values.next().unwrap_or_default(),
                        ),
                    )+
                    prefix,
                }
            }

            /// Replaces this model's parameter definitions.
            #[must_use]
            pub fn with_parameters(mut self, definitions: [ParameterDefinition; simple_model!(@count $($field),+)]) -> Self {
                let mut definitions = definitions.into_iter();
                $(
                    self.$field = definitions.next().unwrap_or_else(|| self.$field.clone());
                )+
                self
            }
        }

        impl Model for $type {
            fn name(&self) -> &str { &self.prefix }

            fn parameter_definitions(&self) -> Vec<ParameterDefinition> {
                vec![$(self.$field.clone()),+]
            }

            fn evaluate(
                &self,
                x: &[f64],
                parameters: &ParameterValues,
                output: &mut [f64],
            ) -> Result<(), FitError> {
                check_output(x, output)?;
                $(let $field = parameters.require(&self.$field.name)?;)+
                for (value, independent) in output.iter_mut().zip(x) {
                    *value = ($body)(*independent, $($field),+);
                }
                ensure_finite(output, concat!($name, " model"))
            }

            fn analytic_jacobian(
                &self,
                x: &[f64],
                _parameters: &ParameterValues,
                parameter_names: &[String],
                output: &mut [f64],
            ) -> Result<bool, FitError> {
                let expected = x.len().saturating_mul(parameter_names.len());
                if output.len() != expected {
                    return Err(FitError::LengthMismatch { x: expected, y: output.len() });
                }
                let definitions = [$(&self.$field),+];
                if definitions.iter().any(|definition| {
                    analytic_column(definition, parameter_names) != AnalyticColumn::Unrelated
                }) {
                    return Ok(false);
                }
                output.fill(0.0);
                Ok(true)
            }


            fn robust_jacobian(
                &self,
                x: &[f64],
                parameters: &ParameterValues,
                parameter_names: &[String],
                output: &mut [f64],
            ) -> Result<bool, FitError> {
                let expected = x.len().saturating_mul(parameter_names.len());
                if output.len() != expected {
                    return Err(FitError::LengthMismatch { x: expected, y: output.len() });
                }
                let columns = [$(&self.$field),+].map(|definition| analytic_column(definition, parameter_names));
                if columns.contains(&AnalyticColumn::UnresolvedBinding) {
                    return Ok(false);
                }
                $(let $field = parameters.require(&self.$field.name)?;)+
                output.fill(0.0);
                for (row, independent) in x.iter().enumerate() {
                    let derivatives = ($derivatives)(*independent, $($field),+);
                    for (column, derivative) in columns.iter().zip(derivatives) {
                        if let AnalyticColumn::Column(column) = column {
                            output[row * parameter_names.len() + column] += derivative;
                        }
                    }
                }
                Ok(output.iter().all(|value| value.is_finite()))
            }

            fn compatibility_component_jacobians(
                &self,
                x: &[f64],
                parameters: &ParameterValues,
                parameter_names: &[String],
                steps: &[f64],
                output: &mut [Vec<f64>],
            ) -> Result<bool, FitError> {
                if output.len() != 1 {
                    return Ok(false);
                }
                compatibility_finite_difference(
                    self,
                    [$(&self.$field),+],
                    x,
                    parameters,
                    parameter_names,
                    steps,
                    &mut output[0],
                )
            }
        }
    };
    (@count $head:ident) => { 1usize };
    (@count $head:ident, $($tail:ident),+) => { 1usize + simple_model!(@count $($tail),+) };
}

simple_model!(
    ConstantModel,
    "A constant background model (`c`).",
    "constant",
    [c],
    |_x: f64, c: f64| c,
    |_x: f64, _c: f64| [1.0]
);
simple_model!(
    LinearModel,
    "A linear background model (`slope * x + intercept`).",
    "linear",
    [slope, intercept],
    |x: f64, slope: f64, intercept: f64| slope * x + intercept,
    |x: f64, _slope: f64, _intercept: f64| [x, 1.0]
);
simple_model!(
    QuadraticModel,
    "A quadratic background model (`a * x^2 + b * x + c`).",
    "quadratic",
    [a, b, c],
    |x: f64, a: f64, b: f64, c: f64| a * x * x + b * x + c,
    |x: f64, _a: f64, _b: f64, _c: f64| [x * x, x, 1.0]
);
simple_model!(
    ExponentialModel,
    "An exponential background model (`amplitude * exp(-x / decay)`).",
    "exponential",
    [amplitude, decay],
    |x: f64, amplitude: f64, decay: f64| amplitude * (-x / decay).exp(),
    |x: f64, amplitude: f64, decay: f64| [
        (-x / decay).exp(),
        amplitude * (-x / decay).exp() * x / decay.powi(2)
    ]
);
simple_model!(
    PowerLawModel,
    "A power-law background model (`amplitude * x^exponent`).",
    "power-law",
    [amplitude, exponent],
    |x: f64, amplitude: f64, exponent: f64| amplitude * x.powf(exponent),
    |x: f64, amplitude: f64, exponent: f64| [
        x.powf(exponent),
        amplitude * x.powf(exponent) * x.ln()
    ]
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnalyticColumn {
    Unrelated,
    Column(usize),
    UnresolvedBinding,
}

fn analytic_column(definition: &ParameterDefinition, parameter_names: &[String]) -> AnalyticColumn {
    if let Some(column) = parameter_names
        .iter()
        .position(|name| name == &definition.name)
    {
        return AnalyticColumn::Column(column);
    }
    match &definition.binding {
        Some(crate::ParameterBinding::EqualTo(source)) => parameter_names
            .iter()
            .position(|name| name == source)
            .map_or(AnalyticColumn::UnresolvedBinding, AnalyticColumn::Column),
        None => AnalyticColumn::Unrelated,
    }
}

fn compatibility_finite_difference<const N: usize>(
    model: &dyn Model,
    definitions: [&ParameterDefinition; N],
    x: &[f64],
    parameters: &ParameterValues,
    parameter_names: &[String],
    steps: &[f64],
    output: &mut [f64],
) -> Result<bool, FitError> {
    let columns = parameter_names.len();
    let expected = x.len().saturating_mul(columns);
    if steps.len() != columns || output.len() != expected {
        return Err(FitError::LengthMismatch {
            x: expected,
            y: output.len(),
        });
    }
    let mappings = definitions
        .iter()
        .map(|definition| analytic_column(definition, parameter_names))
        .collect::<Vec<_>>();
    if mappings.contains(&AnalyticColumn::UnresolvedBinding) {
        return Ok(false);
    }
    output.fill(0.0);
    if mappings
        .iter()
        .all(|mapping| *mapping == AnalyticColumn::Unrelated)
    {
        return Ok(true);
    }
    let mut plus_curve = checked_zeros(x.len())?;
    let mut minus_curve = checked_zeros(x.len())?;
    for column in 0..columns {
        if !mappings.contains(&AnalyticColumn::Column(column)) {
            continue;
        }
        let step = steps[column];
        let mut plus = parameters.clone();
        let mut minus = parameters.clone();
        for (definition, mapping) in definitions.iter().zip(&mappings) {
            if *mapping == AnalyticColumn::Column(column) {
                let baseline = parameters.require(&definition.name)?;
                plus.insert(definition.name.clone(), baseline + step);
                minus.insert(definition.name.clone(), baseline - step);
            }
        }
        model.evaluate(x, &plus, &mut plus_curve)?;
        model.evaluate(x, &minus, &mut minus_curve)?;
        for row in 0..x.len() {
            output[row * columns + column] = (plus_curve[row] - minus_curve[row]) / (2.0 * step);
        }
    }
    Ok(true)
}

fn check_output(x: &[f64], output: &[f64]) -> Result<(), FitError> {
    if x.len() != output.len() {
        return Err(FitError::LengthMismatch {
            x: x.len(),
            y: output.len(),
        });
    }
    Ok(())
}

fn ensure_finite(values: &[f64], model: &str) -> Result<(), FitError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(FitError::NonFinite {
            context: format!("{model} evaluation"),
        })
    }
}

fn checked_zeros(length: usize) -> Result<Vec<f64>, FitError> {
    const MAX_VALUES: usize = 16_777_216;
    if length > MAX_VALUES {
        return Err(FitError::AllocationLimit {
            requested: length,
            limit: MAX_VALUES,
        });
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_allocation_error| FitError::AllocationLimit {
            requested: length,
            limit: MAX_VALUES,
        })?;
    values.resize(length, 0.0);
    Ok(values)
}
