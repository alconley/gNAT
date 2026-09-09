use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ConversationRole {
    User,
    Assistant,
}

impl ConversationRole {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::User => "You",
            Self::Assistant => "Spectrix AI",
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub(crate) struct ConversationMessage {
    pub(crate) role: ConversationRole,
    pub(crate) content: String,
}

impl ConversationMessage {
    pub(crate) fn user(content: impl Into<String>) -> Self {
        Self {
            role: ConversationRole::User,
            content: content.into(),
        }
    }

    pub(crate) fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ConversationRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub(crate) struct AiResponse {
    pub(crate) summary: String,
    #[serde(default)]
    pub(crate) clarification_questions: Vec<String>,
    #[serde(default)]
    pub(crate) notes: Vec<String>,
}

pub(crate) fn parse_model_response(output_text: &str) -> AiResponse {
    match serde_json::from_str::<AiResponse>(output_text) {
        Ok(response) => response,
        Err(error) => AiResponse {
            summary: output_text.trim().to_owned(),
            notes: vec![format!(
                "The model did not return the expected Spectrix JSON format, so the raw answer is shown. Parser detail: {error}"
            )],
            ..AiResponse::default()
        },
    }
}

pub(crate) fn normalize_help_response(mut response: AiResponse, prompt: &str) -> AiResponse {
    response.summary = response.summary.trim().to_owned();
    response.clarification_questions = dedupe_strings(response.clarification_questions);
    response.notes = dedupe_strings(response.notes);

    if let Some(fallback) = source_backed_cut_definition_response(prompt) {
        return fallback;
    }

    if let Some(fallback) = source_backed_gaussian_fit_fallback_response(prompt) {
        return fallback;
    }

    if let Some(fallback) = source_backed_1d_cut_fallback_response(prompt, &response) {
        return fallback;
    }

    if let Some(fallback) = source_backed_2d_cut_fallback_response(prompt, &response) {
        return fallback;
    }

    if prompt_requests_actions(prompt) {
        response.clarification_questions.clear();
        response.summary = "AI action-taking in Spectrix is currently under development. I can still walk you through the manual UI steps for that workflow.".to_owned();
        response.notes.push(
            "For now, the assistant answers usage questions; analysis and direct app changes come later."
                .to_owned(),
        );
    } else if response.summary.is_empty() {
        response.summary = if response.clarification_questions.is_empty() {
            "I did not get a usable answer back from the model. Try rephrasing the question."
                .to_owned()
        } else {
            "I need a little more detail to answer that clearly.".to_owned()
        };
    }

    response.notes = dedupe_strings(response.notes);
    response
}

fn source_backed_cut_definition_response(prompt: &str) -> Option<AiResponse> {
    if !asks_definition(prompt) || prompt_mentions_analysis(prompt) {
        return None;
    }

    let asks_1d = is_1d_cut_question(prompt);
    let asks_2d = is_2d_cut_question(prompt);

    let (summary, notes) = match (asks_1d, asks_2d) {
        (true, false) => {
            (
                "A 1D cut in Spectrix is a one-variable event gate. In the code it is `Cut1D`: a name plus an expression or builder groups, such as `Energy >= 1200 & Energy <= 1300`. Spectrix parses those comparisons and turns them into a Polars boolean mask. Conditions inside one group are ANDed, separate groups are ORed, and separate active cuts are ANDed together.\n\nExperimentally, a 1D cut selects events by one observable: an energy window, a time window, a position window, a PID value, or a validity flag. Events that pass the gate are the ones that contribute to the histogram, fit, or saved filtered parquet output. An interactive 1D histogram cut is the same idea drawn as two draggable vertical bounds on the histogram axis.",
                vec![
                    "This answer is based on the `Cut1D` parsing, mask creation, and active cut filtering code paths.".to_owned(),
                    "Creating a cut is a separate workflow; this answer is the definition and data meaning.".to_owned(),
                ],
            )
        }
        (false, true) => {
            (
                "A 2D cut in Spectrix is a polygon gate drawn on a 2D histogram. In the code it is `Cut2D`: it stores an X column, a Y column, and polygon vertices. When Spectrix applies the cut, it takes each event's `(x, y)` values from those columns and keeps the event only if that point is inside the polygon.\n\nExperimentally, a 2D cut selects a correlated population, such as an energy-position group, a time-energy band, a PID island, or a coincidence band, where a simple one-axis range is not enough.",
                vec![
                    "This answer is based on the `Cut2D`, polygon containment, and active cut filtering code paths.".to_owned(),
                    "Creating a cut is a separate workflow; this answer is the definition and data meaning.".to_owned(),
                ],
            )
        }
        (true, true) => {
            (
                "A 1D cut and a 2D cut are both event gates, but they test different shapes in the data. A 1D cut gates one column with comparisons like `Energy >= low & Energy <= high`; Spectrix parses that into a boolean mask. A 2D cut gates a pair of columns by checking whether each event's `(x, y)` point lies inside a polygon.\n\nExperimentally, use a 1D cut for one observable such as an energy, time, or position window. Use a 2D cut for correlated observables, such as selecting an energy-position group, a time-energy band, a PID island, or a coincidence band on a 2D histogram.",
                vec![
                    "This answer is based on the `Cut1D` and `Cut2D` mask creation and active cut filtering code paths.".to_owned(),
                    "Creating a cut is a separate workflow; this answer is the definition and data meaning.".to_owned(),
                ],
            )
        }
        (false, false) => return None,
    };

    Some(AiResponse {
        summary: summary.to_owned(),
        clarification_questions: Vec::new(),
        notes,
    })
}

fn source_backed_1d_cut_fallback_response(
    prompt: &str,
    response: &AiResponse,
) -> Option<AiResponse> {
    if !is_1d_cut_question(prompt) || prompt_mentions_analysis(prompt) {
        return None;
    }

    if !response_looks_misdirected(response) {
        return None;
    }

    Some(AiResponse {
        summary: "To create a 1D cut, use the Histogram Script cut builder. Open the Histogram Script panel, go to the Cuts section, click `+1D Manual`, name the cut, then click `Builder` on that cut row. Choose the column, comparison operator, and value for the first condition. Use `Add Condition` for another AND condition in the same group, or `Add OR Expression` to make an alternative group. Enable the cut checkbox when you want it applied to histogram generation or filtered parquet saving.\n\nIf you are already looking at a 1D histogram, you can also press `C` in the plot or use the right-click Cuts menu to create a draggable interactive 1D cut. That cut appears under Active Histogram Cuts so it can be enabled and reused.".to_owned(),
        clarification_questions: Vec::new(),
        notes: vec![
            "This answer is based on the 1D cut UI and interactive cut code paths, not the Analysis module.".to_owned(),
        ],
    })
}

fn source_backed_gaussian_fit_fallback_response(prompt: &str) -> Option<AiResponse> {
    if !is_fit_workflow_question(prompt) || prompt_mentions_analysis(prompt) {
        return None;
    }

    Some(AiResponse {
        summary: "Gaussian fitting in Spectrix is manual-first. Place exactly two region markers with `R`, then place every intended peak with `P`, or press `O` to populate editable markers with the original peak finder. Choose the explicit background family in `Fits`; without windows, the selected background is estimated from the region. Optional background windows placed with `B` determine a background that stays fixed during peak fitting. Without windows, peaks and background are fitted jointly. Errors for a fixed-background peak fit exclude background uncertainty. Press `F` to run the fit; detection never launches a fit or chooses a model.\n\nEach fit-results row places Initial and Temp values side by side. Initial values expose editable min/guess/max controls for position, FWHM, and background-subtracted height; fitted Temp measurements use PDG-style scientific notation with their uncertainties. Editing one parameter holds the other displayed variables fixed. The plot shows only a translucent envelope spanning the width/height ranges; position bounds are not drawn. Shared FWHM applies one width to every peak while preserving each height; Independent FWHM permits individual widths. The displayed values and ranges are passed to a worker that polishes the complete solution and uses bounded deterministic recovery when needed. User limits, fixed values, and peak count are preserved. Missing covariance is explained in the fit report rather than replaced with artificial errors.\n\nA successful result becomes the temporary fit and its fitted centers, FWHM values, and heights become the new handles. Fit-quality metrics are advisory and never block `S` or Store Fit. Press `S` or Store Fit to keep it. Stored fits are not changed by later handle edits.".to_owned(),
        clarification_questions: Vec::new(),
        notes: vec![
            "This answer is based on the 1D histogram keybinds, Gaussian fitter, Fit Panel, and calibration code paths.".to_owned(),
            "If a quadratic calibration is not safely invertible over the relevant range, Spectrix warns and falls back to the raw X axis for display.".to_owned(),
        ],
    })
}

fn source_backed_2d_cut_fallback_response(
    prompt: &str,
    response: &AiResponse,
) -> Option<AiResponse> {
    if !is_2d_cut_question(prompt) || prompt_mentions_analysis(prompt) {
        return None;
    }

    if !response_looks_misdirected(response) {
        return None;
    }

    let summary = if asks_definition(prompt) {
        "A 2D cut in Spectrix is a polygon gate drawn on a 2D histogram. In the code it is `Cut2D`: it stores an X column, a Y column, and polygon vertices. When Spectrix applies the cut, it takes each event's `(x, y)` values from those columns and keeps the event only if that point is inside the polygon."
    } else {
        "To create a 2D cut, first create or open a 2D histogram so Spectrix knows the X and Y columns. In the 2D plot, press `C`, or right-click the plot and choose `Cuts` then `+`. Click points on the plot to add polygon vertices, then double-click to finish the polygon. After it exists, drag vertices to edit it. The cut name defaults to something like `Y v X Cut 1`, and the X/Y columns come from the 2D histogram axes. Use `Save` in the cut menu if you want to write it to JSON for reuse."
    };

    Some(AiResponse {
        summary: format!(
            "{summary}\n\nOne important UI distinction: the Histogram Script `+2D` button loads an already-saved 2D cut JSON. Drawing a brand-new 2D cut happens from a 2D histogram view."
        ),
        clarification_questions: Vec::new(),
        notes: vec![
            "Unsaved 2D cuts created on a plot can still appear under Active Histogram Cuts and can be enabled for histogram generation or parquet filtering.".to_owned(),
            "This answer is based on the `Cut2D`, 2D histogram context menu, and polygon interaction code paths.".to_owned(),
        ],
    })
}

fn response_looks_misdirected(response: &AiResponse) -> bool {
    let combined = format!(
        "{}\n{}\n{}",
        response.summary,
        response.clarification_questions.join("\n"),
        response.notes.join("\n")
    )
    .to_lowercase();

    !response.clarification_questions.is_empty()
        || combined.contains("analysis module")
        || combined.contains("simulation")
        || combined.contains("post-processing")
        || combined.contains("feature description for 'analysis'")
        || combined.contains("correlation tools")
        || combined.contains("data viewer")
        || combined.contains("rectangular or elliptical")
        || combined.contains("cut-planes")
        || combined.contains("2d or 3d plot")
}

fn is_1d_cut_question(prompt: &str) -> bool {
    let normalized = prompt.to_lowercase();
    let mentions_cut = normalized.contains("cut") || normalized.contains("gate");
    let mentions_1d = normalized.contains("1d") || normalized.contains("one d");

    mentions_cut && mentions_1d
}

fn is_fit_workflow_question(prompt: &str) -> bool {
    let normalized = prompt.to_lowercase();
    let mentions_fit = normalized.contains("fit")
        || normalized.contains("fits")
        || normalized.contains("fitting")
        || normalized.contains("fitted");
    let mentions_gaussian = normalized.contains("gaussian");
    let mentions_peak = normalized.contains("peak");
    let mentions_marker = normalized.contains("marker");
    let mentions_workflow = normalized.contains("how do")
        || normalized.contains("how can")
        || normalized.contains("procedure")
        || normalized.contains("steps")
        || normalized.contains("perform")
        || normalized.contains("do fits")
        || normalized.contains("make fits")
        || normalized.contains("fit stuff")
        || normalized.contains("run fits")
        || normalized.contains("run a fit")
        || normalized.contains("make a fit");

    mentions_fit && (mentions_gaussian || mentions_peak || mentions_marker || mentions_workflow)
}

fn is_2d_cut_question(prompt: &str) -> bool {
    let normalized = prompt.to_lowercase();
    let mentions_cut = normalized.contains("cut") || normalized.contains("gate");
    let mentions_2d = normalized.contains("2d") || normalized.contains("two d");

    mentions_cut && mentions_2d
}

fn asks_definition(prompt: &str) -> bool {
    let normalized = prompt.trim().to_lowercase();
    normalized.starts_with("what is")
        || normalized.starts_with("what are")
        || normalized.starts_with("explain")
        || normalized.starts_with("define")
}

fn prompt_mentions_analysis(prompt: &str) -> bool {
    let normalized = prompt.to_lowercase();
    normalized.contains("analysis")
        || normalized.contains("se-sps")
        || normalized.contains("cross section")
        || normalized.contains("cross-section")
}

pub(crate) fn format_assistant_message(response: &AiResponse) -> String {
    let mut message = response.summary.trim().to_owned();

    if !response.clarification_questions.is_empty() {
        if !message.is_empty() {
            message.push_str("\n\n");
        }
        message.push_str("Clarification needed:");
        for question in &response.clarification_questions {
            message.push_str("\n- ");
            message.push_str(question.trim());
        }
    }

    if !response.notes.is_empty() {
        if !message.is_empty() {
            message.push_str("\n\n");
        }
        message.push_str("Notes:");
        for note in &response.notes {
            message.push_str("\n- ");
            message.push_str(note.trim());
        }
    }

    message
}

pub(crate) fn response_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["summary", "clarification_questions", "notes"],
        "properties": {
            "summary": { "type": "string" },
            "clarification_questions": {
                "type": "array",
                "items": { "type": "string" }
            },
            "notes": {
                "type": "array",
                "items": { "type": "string" }
            }
        }
    })
}

fn prompt_requests_actions(prompt: &str) -> bool {
    let normalized = prompt.trim().to_lowercase();
    if normalized.is_empty() {
        return false;
    }

    let question_starters = [
        "how do",
        "how can",
        "what is",
        "what are",
        "where is",
        "where do",
        "why does",
        "can i",
        "could i",
        "does spectrix",
        "is there",
    ];
    if normalized.contains('?')
        || question_starters
            .iter()
            .any(|starter| normalized.starts_with(starter))
    {
        return false;
    }

    let action_starters = [
        "build ",
        "create ",
        "make ",
        "generate ",
        "add ",
        "apply ",
        "set up ",
        "configure ",
        "fit ",
        "plot ",
        "save ",
        "combine ",
        "filter ",
        "do ",
    ];

    action_starters
        .iter()
        .any(|starter| normalized.starts_with(starter))
}

fn dedupe_strings(values: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .filter(|value| seen.insert(value.to_lowercase()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_request_is_kept_help_only() {
        let response = normalize_help_response(
            AiResponse {
                summary: "Done".to_owned(),
                ..AiResponse::default()
            },
            "create a histogram for me",
        );

        assert!(response.summary.contains("under development"));
        assert!(response.clarification_questions.is_empty());
    }

    #[test]
    fn json_response_formats_as_conversation_message() {
        let raw = r#"{
            "summary": "Use Get Files/Directory first.",
            "clarification_questions": [],
            "notes": ["Then open Histogram Script."]
        }"#;
        let response = parse_model_response(raw);
        let message = format_assistant_message(&response);

        assert!(message.contains("Use Get Files/Directory first."));
        assert!(message.contains("Then open Histogram Script."));
    }

    #[test]
    fn misdirected_analysis_answer_is_repaired_for_1d_cut_question() {
        let response = normalize_help_response(
            AiResponse {
                summary: "The Analysis module is primarily for post-processing.".to_owned(),
                clarification_questions: vec!["Which analysis calculation do you mean?".to_owned()],
                notes: vec!["The feature description for 'Analysis' is sparse.".to_owned()],
            },
            "How do I create a 1D cut?",
        );

        assert!(response.summary.contains("+1D Manual"));
        assert!(response.summary.contains("Builder"));
        assert!(response.clarification_questions.is_empty());
        assert!(!response.summary.contains("Analysis module is primarily"));
    }

    #[test]
    fn creation_answer_is_replaced_for_1d_cut_definition_question() {
        let response = normalize_help_response(
            AiResponse {
                summary: "To create a 1D cut, open Histogram Script and click +1D Manual."
                    .to_owned(),
                clarification_questions: Vec::new(),
                notes: Vec::new(),
            },
            "What is a 1D cut?",
        );

        assert!(response.summary.starts_with("A 1D cut in Spectrix"));
        assert!(response.summary.contains("one-variable event gate"));
        assert!(response.summary.contains("Cut1D"));
        assert!(response.summary.contains("Polars boolean mask"));
        assert!(response.summary.contains("ANDed"));
        assert!(response.summary.contains("ORed"));
        assert!(response.summary.contains("one observable"));
        assert!(!response.summary.starts_with("To create"));
        assert!(response.clarification_questions.is_empty());
    }

    #[test]
    fn generic_correlation_answer_is_repaired_for_2d_cut_creation_question() {
        let response = normalize_help_response(
            AiResponse {
                summary: "To create 2D cuts, use the data viewer's correlation tools and select a rectangular or elliptical region.".to_owned(),
                clarification_questions: Vec::new(),
                notes: vec![
                    "The process depends on the specific version of the software.".to_owned(),
                ],
            },
            "How can I create a 2D cut?",
        );

        assert!(response.summary.contains("press `C`"));
        assert!(response.summary.contains("double-click"));
        assert!(response.summary.contains("`+2D` button loads"));
        assert!(response.clarification_questions.is_empty());
    }

    #[test]
    fn gaussian_fit_question_gets_source_backed_fallback_when_answer_is_misdirected() {
        let response = normalize_help_response(
            AiResponse {
                summary: "Use the general analysis workflow to post-process the data.".to_owned(),
                clarification_questions: vec!["Which analysis mode do you mean?".to_owned()],
                notes: vec!["This depends on the analysis module.".to_owned()],
            },
            "How do I fit Gaussian peaks?",
        );

        assert!(response.summary.contains("manual-first"));
        assert!(response.summary.contains("region markers with `R`"));
        assert!(response.summary.contains("peak with `P`"));
        assert!(
            response
                .summary
                .contains("background windows placed with `B`")
        );
        assert!(response.summary.contains("Press `F`"));
        assert!(response.summary.contains("translucent envelope"));
        assert!(response.summary.contains("press `O`"));
        assert!(response.summary.contains("temporary fit"));
        assert!(response.clarification_questions.is_empty());
    }

    #[test]
    fn generic_fit_workflow_question_gets_source_backed_fit_procedure() {
        let response = normalize_help_response(
            AiResponse {
                summary: "To perform fits in Spectrix, first create and display a 1D histogram. With your cursor inside the histogram plot area, press `F` to initiate a Gaussian fit.".to_owned(),
                clarification_questions: Vec::new(),
                notes: Vec::new(),
            },
            "how do i do fits?",
        );

        assert!(response.summary.contains("manual-first"));
        assert!(response.summary.contains("region markers with `R`"));
        assert!(response.summary.contains("peak with `P`"));
        assert!(
            response
                .summary
                .contains("background windows placed with `B`")
        );
        assert!(response.summary.contains("Press `F`"));
        assert!(response.summary.contains("Press `S`"));
        assert!(response.clarification_questions.is_empty());
    }

    #[test]
    fn clarification_is_repaired_for_2d_cut_definition_question() {
        let response = normalize_help_response(
            AiResponse {
                summary: "Please specify if you mean visualizing data or filtering data."
                    .to_owned(),
                clarification_questions: vec![
                    "Are you looking to create a 2D or 3D plot?".to_owned(),
                ],
                notes: Vec::new(),
            },
            "what is a 2d cut?",
        );

        assert!(response.summary.contains("polygon gate"));
        assert!(response.summary.contains("Cut2D"));
        assert!(response.summary.contains("inside the polygon"));
        assert!(response.clarification_questions.is_empty());
    }
}
