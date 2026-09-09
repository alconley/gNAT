use super::codebase::CodebaseContext;
use super::context::AiContextSnapshot;
use super::response::{ConversationMessage, ConversationRole};

const DEFAULT_SYSTEM_PROMPT: &str = r#"
You are Spectrix's in-app help assistant.

Your job is to answer questions about how to use Spectrix as it exists today. Return only valid JSON that matches the provided schema.

Response rules:
- Put the main answer in `summary`.
- Use `clarification_questions` only if a short follow-up is required to answer correctly.
- Use `notes` for short extra tips, caveats, or UI labels that may help the user.

Behavior rules:
- Before answering, use the provided codebase investigation packet: matched files, UI controls, interaction clues, call-chain clues, data/experiment operation clues, and source excerpts.
- Treat the codebase investigation as an implementation-to-UI trace. First identify the code/function that implements what the user is asking about. Then follow where that function is called, and keep following callers and UI controls until you can describe how the user reaches it in Spectrix.
- Reconstruct the user's workflow by tracing UI labels and entry points in order. Prefer real labels from the UI clues over generic wording.
- Be explicit about how the user interacts with the UI. If the source shows a specific interaction, say it precisely: `click`, `right-click`, `double-click`, `hover`, `press` a key, `scroll`, `click and drag`, or `drag`.
- Do not collapse a specific gesture into generic wording. For example, if the source shows a context menu, say `right-click`; if it shows draggable handles or regions, say `click and drag`; if it uses hover behavior, say `hover`.
- Only mention a specific interaction style when the codebase evidence supports it.
- When interaction clues or call-chain clues are available, use them to connect implementation details back to menus, buttons, panels, context menus, keybinds, plot gestures, and hover states.
- Explain what the code does to the data when the user asks "what is", "why", "what happens", or any experiment/data question.
- For "what is" questions, lead with the concept and experiment/data meaning before giving UI steps.
- Interpret Spectrix as an experimental event-data analysis tool. In experiments such as nuclear physics, users are often interested in observables like energy, time, position, PID, or calibrated derived values. Histograms are binned event counts; cuts/gates are event-selection masks; fits extract peak centroids, widths, areas/yields; calibration maps detector/channel observables to physical units where supported.
- When the source shows that a histogram can be filled from multiple selected columns or source pairs, explain that behavior explicitly instead of implying there is only one backing column.
- Trust implementation source excerpts and UI/data clues over docs, prior assumptions, or generic app behavior.
- If the codebase investigation did not find enough evidence, say what was missing and answer only what the source supports.
- Do not switch to the Analysis module unless the user asks about Analysis, SE-SPS, cross sections, or post-fit analysis.
- Do not ask clarification when the source excerpts already show the likely workflow. For broad "how do I..." questions, explain the common path and mention alternatives.
- The current user request is authoritative. Do not reuse a previous answer topic if the current request asks about a different Spectrix concept.
- If the user asks how to do something in Spectrix, answer with practical UI steps that are simple, direct, detailed, and explicit about the interaction type.
- Refer to real Spectrix UI names when helpful, such as Processor, Histogram Script, Variables, Column Creation, Column Groups, 1D Cuts, Builder, Output Name, Calculate Histograms, Selected File Settings, fitting controls, and active cuts.
- If the user asks you to create, apply, edit, calculate, configure, analyze, or otherwise do something for them, clearly say that AI action-taking and AI analysis in Spectrix are currently under development.
- If Spectrix does not currently support what the user wants, say so plainly.
- Keep answers concise, concrete, and focused on the current app.
"#;

const MAX_HISTORY_MESSAGES: usize = 8;

pub(crate) fn default_system_prompt() -> &'static str {
    DEFAULT_SYSTEM_PROMPT.trim()
}

pub(crate) fn build_user_prompt(
    prompt: &str,
    conversation_history: &[ConversationMessage],
    snapshot: &AiContextSnapshot,
) -> Result<String, String> {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return Err("Ask a question about how to use Spectrix.".to_owned());
    }

    let codebase_context = CodebaseContext::for_query(trimmed);

    Ok(format!(
        "Current user request:\n{trimmed}\n\nConversation so far:\n{}\n\nCurrent Spectrix session state:\n{}\n\n{}\n\n{}\n\nExperimental-data interpretation guide:\n{}\n\nAnswering reminder:\n- First identify what feature the user is asking about from the matched source files.\n- Find the implementation code/function that handles it.\n- Then trace where that function is called; keep following callers, UI controls, context menus, keybinds, hover behavior, and draggable plot elements until you can explain how the user reaches it in the app.\n- If the user asks what something is, answer the definition and experiment/data meaning first.\n- Then give simple, direct, detailed UI steps using exact Spectrix UI names.\n- Name the interaction method precisely when the source supports it: click, right-click, double-click, hover, press a key, or click and drag.\n- Then explain what the code does to the user's experimental data when relevant.\n- Do not answer about the Analysis module unless the request is actually about Analysis.\n- Do not claim you changed Spectrix state or performed analysis.",
        conversation_history_section(conversation_history),
        snapshot.to_prompt_section(),
        source_derived_workflow_hints(trimmed),
        codebase_context.to_prompt_section(),
        experimental_data_interpretation_guide(),
    ))
}

fn experimental_data_interpretation_guide() -> &'static str {
    "- Parquet columns represent event-by-event experimental observables, such as energy, time, position, detector IDs, or calibrated derived values.\n- Histogram Script column groups can alias one detector/source family name to many real parquet columns.\n- A histogram bins one or two observables and displays counts/yields per bin.\n- Spectrix can fill the same histogram from multiple selected source columns, column-group aliases, or vetted `(x, y)` source pairs when the Histogram Script configuration asks for it.\n- Grouped computed-column expressions can expand into many derived output columns instead of only one.\n- A cut/gate creates a boolean event-selection mask. A 1D cut compares one column to bounds or values. A 2D cut tests whether an event's `(x, y)` point lies inside a polygon gate.\n- Applying cuts before histogramming means only events passing the enabled gates contribute to the displayed histogram, fit, or saved filtered parquet file.\n- Gaussian fitting measures peak centroids, widths, amplitudes, and areas. In experiments such as nuclear physics, those are commonly used for energy/time/position calibration checks, resolution estimates, and yield extraction.\n- Calibration code maps detector/channel coordinates to physical units where the source code supports it.\n- Cross-section analysis, where present, combines fitted yields with beam/current/target/solid-angle style metadata; do not imply it is available for unrelated workflows unless source excerpts show it."
}

fn source_derived_workflow_hints(prompt: &str) -> String {
    let mut sections = Vec::new();

    if is_histogram_script_question(prompt) {
        sections.push(
            r#"Source-derived workflow hints for Histogram Script setup:
- The General Histogram Script tab exposes `Variables`, `Column Creation`, `Column Groups`, and `Cuts` sections.
- `Variables`, `Column Creation`, and `Column Groups` rows can be reordered by dragging the `↕` handle.
- Searchable pickers in Histogram Script use natural numeric sorting, so detector names such as `Ring0`, `Ring1`, `Ring10` stay in the expected order.
- Picker search boxes support normal substring matching and glob-style filters such as `S1Ring*Time` or `S1Ring?Time`.
- Multi-select pickers keep selected entries listed at the top and expose `Select All` and `Clear All` actions.
- `Column Groups` lets one alias stand for many real columns, which is useful for detector arrays or segmented/ring detectors.
- Column-group aliases appear in histogram source pickers and anywhere the `Column Creation` builder accepts a source column.
- In `Column Creation`, grouped expressions expand into multiple computed columns instead of only one.
- If several group aliases appear in the same computed-column expression, Spectrix expands all combinations.
- The builder preview tells the user how many grouped columns will be created and shows example generated names.
- The `Output Name` field accepts `{}` to place grouped source column names at a specific point in the generated alias; if `{}` is omitted, Spectrix appends those grouped source column names automatically.
"#,
        );
    }

    if is_fit_workflow_question(prompt) {
        sections.push(
            r#"Source-derived workflow hints for Gaussian fitting:
- Gaussian fitting is a 1D histogram workflow. First move the cursor into the 1D histogram you want to fit, because the fitting keybinds act at the current cursor position inside that plot.
- The default background model is `None`. If you want background fitting, open the 1D histogram right-click menu, open `Fits`, and choose a background model such as `Linear`, `Quadratic`, `Power Law`, or `Exponential`.
- Marker keybinds: `R` adds a region marker, `P` adds a peak marker, `B` adds a background marker pair, `-` removes the nearest marker, and `Delete` clears temporary markers and temp fits.
- Press `R` twice to place the two region markers that define the fit interval. `R` keeps at most two active region markers, so adding another region marker after two are present clears the old pair and starts again.
- Press `B` to add a green background marker pair at the cursor. It starts as a very narrow region, and you can click and drag the green marker lines to widen or narrow it. The bins between each background pair are the points used when Spectrix fits just the background with `G`.
- Press `G` to estimate only the selected background model in a worker and make it available for locking. It uses background windows when present, or the selected region when no windows are supplied.
- Press `P` to add every peak marker manually. Gaussian fitting requires at least one explicit peak marker and exactly two region markers.
- Press `F` to fit the marked Gaussians with Spectrix's native Rust engine in a worker. It polishes the complete solution and uses bounded deterministic recovery when convergence or covariance checks fail.
- Choose the background family yourself. With no background windows, it is estimated automatically from the region. Supplied windows determine the background, which stays fixed during the peak fit; overlapping windows count each bin once. These peak errors are conditional on that background. Without windows, background and peaks are fitted jointly unless Lock manual background holds an explicit background fit fixed.
- The fit-results table keeps editable Initial min/guess/max controls and fitted Temp results on the same peak row. Temp values use PDG-style scientific notation with their uncertainties. Editing one guess holds the other displayed variables fixed. The plot shows only a translucent width/height envelope; position bounds are edited in the table and are not drawn.
- `O` runs the original peak finder to populate editable manual peak markers. It never launches a fit or chooses a model.
- The fit settings include Shared/Independent FWHM, Free Position, automatic re-estimation after moving a peak, background lock/refine coupling, and fit-line display toggles. Per-peak initial FWHM bounds control each fit's width range.
- After a successful fit, the result appears as the `Temp` fit and its fitted position, FWHM, and height become the new handles.
- Press `S` / `Store Fit` when you want to keep the temp fit as a stored fit for later comparison, calibration, saving, or refitting.
- In the Fit Panel you can assign per-peak UUID values and energies. `UUID = 0` is the default invalid UUID. `Energy = -1` means that peak is invalid for calibration.
- UUID labels can be drawn above peaks and adjusted in the Fit Panel with `Size`, `Lift`, and `Guide`.
- If you enable `Calibration` in the Fit Panel, you can type the coefficients manually and click `Calibrate`, or fit them from assigned energies with `Linear` or `Quadratic`.
- Applying calibration updates all stored fits and the temp fit directly in Rust.
- Use `Save Fits` / `Load Fits` for Spectrix JSON workflows. Legacy JSON remains loadable, but the old Python model-result payload is ignored and omitted from new saves.
- After calibration, the Fit Panel also shows the calibration curve and residuals. If a quadratic calibration is not safely invertible over the relevant range, Spectrix warns and falls back to the raw X axis for display.
"#,
        );
    }

    if is_calibration_question(prompt) {
        sections.push(
            r#"Source-derived workflow hints for calibration:
- These calibration workflow hints also apply when the user is asking about gain matching or shifting columns, because the same coefficient-based transforms are used there.
- Spectrix has two different calibration workflows: calibrating Gaussian fit results/display on an existing 1D histogram, and creating a new calibrated event-data column in Histogram Script.
- For 1D histogram fit calibration, first fit and store Gaussian peaks. In the Fit Panel table, enter each peak's assigned energy and uncertainty; peaks with assigned energy `-1` are ignored as invalid calibration points.
- In the Fit Panel, enable `Calibration`. You can type coefficients directly as `a`, `b`, and `c` for `a*x^2 + b*x + c`, then click `Calibrate` to apply them to stored/temp fits and the bin edges on the histogram.
- To derive coefficients from stored fits, use `Linear` for a linear calibration from at least 2 valid calibration points, or `Quadratic` for a quadratic calibration from at least 3 distinct valid points.
- Fit calibration affects the 1D histogram display/fit results when the calibration is safe over the histogram range. It attaches calibrated mean, sigma, and FWHM values to Gaussian parameters, while area and amplitude are copied through.
- Fit calibration does not create a new parquet/event column. It is for calibrated fit quantities and calibrated display behavior on that histogram.
- Applying calibration updates all stored fits and the temp fit, so large fit collections can take a moment.
- To create a calibrated event-data column in Histogram Script, there are two main paths.
- Path 1: use the dedicated `Calibration` section under the General Histogram Script tab. Add a source column, fill in `A`, `B`, and `C` for `output = A*input^2 + B*input + C`, and optionally edit `Output Column Name`.
- In that Calibration table, the output name defaults to the source column name. Leaving it unchanged overwrites that column name in Histogram Script outputs; changing it creates a separate calibrated column. Rows only apply when the coefficients differ from the defaults `0, 1, 0`.
- The Calibration section also supports `Import [.csv]` and `Export [.csv]` for `column_name,a,b,c,output_column_name`. Rows whose source columns are missing from the current parquet schema stay visible in red and are skipped until those columns exist again.
- Path 2: use `Column Creation` -> `+` -> `Builder` if you want to assemble the calibration manually from reusable `Variables` or more general expression terms.
- In Column Creation, a quadratic calibrated column can be built as three terms: coefficient `a` times source column with power `2`, plus coefficient `b` times source column with power `1`, plus constant `c`. The coefficients can be literal values or Variables.
- Column Creation also accepts `Column Groups` aliases. Grouped computed-column expressions expand into multiple outputs, and if several group aliases are used in the same expression, Spectrix expands all combinations.
- The grouped-column builder preview shows how many outputs will be created and example generated names.
- Use `Output Name` with `{}` if you want grouped source column names inserted at a specific location in the generated aliases. If `{}` is omitted, Spectrix appends the grouped source column names automatically.
- Give the computed column an alias such as `ColumnNameEnergyCalibrated`. Computed-column aliases are sanitized to letters, numbers, underscores, and optionally `{}` for grouped output naming.
- Derived/calibrated columns can then be selected like native columns in 1D/2D histogram definitions and cuts. To view the calibrated data as a new 1D histogram, add a `+1D` histogram using the calibrated-column alias, set calibrated range/bins, and calculate histograms.
"#,
        );
    }

    if is_2d_projection_question(prompt) {
        sections.push(
            r#"Source-derived workflow hints for 2D projections:
- 2D projections are created from an open 2D histogram. The cursor must be inside the plot for projection keybinds to act.
- Press `X` to toggle an X projection. It sums counts over a selected Y interval and opens a 1D histogram named `X-Projection of ...` with the 2D histogram's X column as its source column.
- Press `Y` to toggle a Y projection. It sums counts over a selected X interval and opens a 1D histogram named `Y-Projection of ...` with the 2D histogram's Y column as its source column.
- Projection regions initialize from the current visible axis bounds with a small inset from each edge.
- Drag a projection boundary line by its center dot to adjust the selected interval. Drag the filled span between projection lines to translate the whole projection window while keeping its width.
- You can also right-click the 2D plot and open `Projections` to toggle `Add X Projection` or `Add Y Projection`, edit X1/X2 or Y1/Y2 values, or set the projection span by width/range.
- The projection windows are normal 1D histogram windows, so 1D controls such as fitting, peak markers, statistics, rebinning, and 1D cuts can be used on the projected spectrum.
- 1D cuts created on projection windows inherit the projected axis column name automatically.
"#,
        );
    }

    if is_histogram_setup_question(prompt) {
        sections.push(
            r#"Source-derived workflow hints for histogram source selection:
- In Histogram Script, 1D histogram rows use searchable multi-select source-column pickers.
- Those pickers support natural numeric ordering, glob-style filters such as `S1Ring*Time`, and `Select All` / `Clear All` actions.
- Selecting multiple 1D source columns fills the same histogram once per selected column.
- 1D histogram sources can also be `Column Groups` aliases, so one source row can resolve to many detector columns.
- If a 1D histogram name contains `{}`, Spectrix creates one histogram per resolved grouped source and inserts the detected token into the name. Without `{}`, it fills the same histogram once per resolved source.
- If a 1D histogram is filled from more than one source column or configuration, Spectrix clears that histogram's stored source-column label instead of leaving a misleading single column name behind.
- In Histogram Script, 2D histogram rows use searchable multi-select pickers for both X and Y source columns.
- 2D X and Y pickers can also resolve `Column Groups` aliases.
- Spectrix expands those selections into explicit `(x, y)` source pairs before filling the histogram.
- It skips self-pairs where `x` and `y` resolve to the same column.
- It also suppresses mirrored duplicates such as `(A, B)` and `(B, A)` so the same detector pairing is not double-counted.
- If a 2D histogram name contains `{}`, Spectrix uses the X token, Y token, or combined `X_Y` tokens when grouped sources are expanded.
- If a 2D histogram is filled from more than one X/Y source pair, Spectrix clears the stored X/Y source labels for that histogram.
- 2D cuts are only available when a 2D histogram has one unambiguous X/Y source pair. For multi-source 2D histograms, plot-created 2D cuts are disabled and existing ones are not exposed as active histogram cuts.
"#,
        );
    }

    if is_1d_cut_question(prompt) {
        if asks_definition(prompt) {
            sections.push(
                r#"Source-derived concept hints for 1D cuts:
- In Spectrix, a 1D cut is a one-variable gate: it keeps or rejects event rows by comparing numeric column values to one or more limits or literal values.
- In the code it is `Cut1D`. It stores a name, an expression string, an active flag, builder groups, and parsed condition groups.
- Conditions inside one builder group are combined with AND. Separate builder groups are combined with OR.
- When applied, Spectrix turns those parsed conditions into Polars boolean masks. Conditions in a group are ANDed, groups are ORed, and separate active cuts are combined with AND.
- Experimentally, a 1D cut is how the user gates on one event observable, such as an energy window, time window, position window, PID value, or validity flag, so only matching events contribute to histograms, fits, or saved filtered parquet output.
- An interactive 1D cut drawn from a 1D histogram is the same idea: two draggable vertical bounds are synced into an expression like `column >= low & column <= high`.
"#
            );
        } else {
            sections.push(
                r#"Source-derived workflow hints for 1D cuts:
- A 1D cut is a one-column event gate. It compares event values in a selected column against bounds or literal values, then uses the resulting boolean mask when calculating histograms or saving filtered parquet.
- The Histogram Script cut UI creates manual 1D cuts with the `+1D Manual` button and loads saved 1D cuts with `+1D Load`.
- A new manual 1D cut starts in Builder mode. In the 1D cut row, click `Builder` to open the dedicated editor.
- In the builder, each expression group combines conditions with AND. `Add Condition` adds another AND condition to the current group.
- `Add OR Expression` starts another expression group, so separate groups are combined with OR.
- Enable the cut checkbox when you want it used by histogram generation or parquet filtering.
- From an existing 1D histogram, pressing `C` or using the right-click Cuts menu can create an interactive draggable 1D cut if the histogram has a source column. Those appear under Active Histogram Cuts.
"#,
            );
        }
    }

    if is_2d_cut_question(prompt) {
        sections.push(
            r#"Source-derived workflow hints for 2D cuts:
- In Spectrix, a 2D cut is a graphical polygon gate tied to an X column and a Y column. In the code it is `Cut2D`, which stores polygon vertices and checks whether each row's `(x, y)` point is inside the polygon.
- To create a new 2D cut, first create/open a 2D histogram so Spectrix knows the X/Y columns.
- 2D cuts are only available when that 2D histogram resolves to one unambiguous X/Y source pair. If the histogram was filled from multiple source pairs, Spectrix clears the stored X/Y source labels and disables plot-created 2D cuts for that plot.
- In the 2D plot, press `C` or right-click the plot and use `Cuts` -> `+` to start a new polygon cut.
- Click on the 2D plot to add polygon vertices. Double-click to finish adding vertices.
- After creation, drag vertices to edit the polygon.
- The cut name defaults to `Y v X Cut N`; the X/Y column names come from the current 2D histogram axes. Verify them in the Cuts menu/info.
- Use `Save` in the cut menu to write the 2D cut to JSON for reuse.
- The Histogram Script `+2D` button loads a saved 2D cut JSON. It is not the main path for drawing a new blank 2D polygon.
- Unsaved 2D cuts from a plot can still appear under Active Histogram Cuts and can be enabled for histogram generation or parquet filtering.
"#
        );
    }

    if sections.is_empty() {
        "Source-derived workflow hints: no special workflow hint matched; rely on the source excerpts below.".to_owned()
    } else {
        sections.join("\n")
    }
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

fn is_calibration_question(prompt: &str) -> bool {
    let normalized = prompt.to_lowercase();

    let mentions_calibration = normalized.contains("calibrat");
    let mentions_gain_matching = normalized.contains("gain match")
        || normalized.contains("gain-match")
        || normalized.contains("gain matching");
    let mentions_shifting = normalized.contains("shifting")
        || normalized.contains("shift column")
        || normalized.contains("shift columns")
        || normalized.contains("energy shift")
        || normalized.contains("offset");

    mentions_calibration || mentions_gain_matching || mentions_shifting
}

fn is_2d_projection_question(prompt: &str) -> bool {
    let normalized = prompt.to_lowercase();
    let mentions_projection = normalized.contains("projection") || normalized.contains("project");
    let mentions_2d = normalized.contains("2d") || normalized.contains("two d");
    let mentions_histogram = normalized.contains("histogram") || normalized.contains("hist");

    mentions_projection && (mentions_2d || mentions_histogram)
}

fn is_1d_cut_question(prompt: &str) -> bool {
    let normalized = prompt.to_lowercase();
    let mentions_cut = normalized.contains("cut") || normalized.contains("gate");
    let mentions_1d = normalized.contains("1d") || normalized.contains("one d");

    mentions_cut && mentions_1d
}

fn is_2d_cut_question(prompt: &str) -> bool {
    let normalized = prompt.to_lowercase();
    let mentions_cut = normalized.contains("cut") || normalized.contains("gate");
    let mentions_2d = normalized.contains("2d") || normalized.contains("two d");

    mentions_cut && mentions_2d
}

fn is_histogram_setup_question(prompt: &str) -> bool {
    let normalized = prompt.to_lowercase();
    let mentions_histogram = normalized.contains("histogram") || normalized.contains("hist");
    let mentions_sources = normalized.contains("column")
        || normalized.contains("columns")
        || normalized.contains("fill")
        || normalized.contains("source")
        || normalized.contains("pair")
        || normalized.contains("pairs")
        || normalized.contains("1d")
        || normalized.contains("2d")
        || normalized.contains("axis");

    mentions_histogram && mentions_sources
}

fn is_histogram_script_question(prompt: &str) -> bool {
    let normalized = prompt.to_lowercase();

    normalized.contains("histogram script")
        || normalized.contains("column creation")
        || normalized.contains("computed column")
        || normalized.contains("computed columns")
        || normalized.contains("column group")
        || normalized.contains("column groups")
        || normalized.contains("group alias")
        || normalized.contains("output name")
        || normalized.contains("combo box")
        || normalized.contains("picker")
        || normalized.contains("glob")
        || normalized.contains("builder")
}

fn asks_definition(prompt: &str) -> bool {
    let normalized = prompt.trim().to_lowercase();
    normalized.starts_with("what is")
        || normalized.starts_with("what are")
        || normalized.starts_with("explain")
        || normalized.starts_with("define")
}

fn conversation_history_section(conversation_history: &[ConversationMessage]) -> String {
    if conversation_history.is_empty() {
        return "No previous messages in this AI conversation.".to_owned();
    }

    let start = conversation_history
        .len()
        .saturating_sub(MAX_HISTORY_MESSAGES);
    conversation_history[start..]
        .iter()
        .map(|message| match message.role {
            ConversationRole::User => format!("User: {}", message.content.trim()),
            ConversationRole::Assistant => format!("Assistant: {}", message.content.trim()),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::context::AiContextSnapshot;

    fn empty_snapshot() -> AiContextSnapshot {
        AiContextSnapshot {
            selected_files: Vec::new(),
            loaded_columns: Vec::new(),
            existing_derived_columns: Vec::new(),
            general_cut_names: Vec::new(),
            active_histogram_cut_names: Vec::new(),
            existing_histograms: Vec::new(),
        }
    }

    #[test]
    fn user_prompt_includes_source_code_context() {
        let prompt = build_user_prompt("How do I create a 1D cut?", &[], &empty_snapshot())
            .expect("prompt should build");

        let workflow_index = prompt
            .find("Source-derived workflow hints for 1D cuts")
            .expect("workflow hints should be included");
        let codebase_index = prompt
            .find("Codebase search results")
            .expect("codebase search should be included");

        assert!(workflow_index < codebase_index);
        assert!(prompt.contains("+1D Manual"));
        assert!(prompt.contains("Add OR Expression"));
        assert!(prompt.contains("Current Spectrix session state"));
        assert!(prompt.contains("Experimental-data interpretation guide"));
        assert!(prompt.contains("hover behavior"));
        assert!(
            prompt.contains(
                "click, right-click, double-click, hover, press a key, or click and drag"
            )
        );
        assert!(!prompt.contains("## Analysis"));
        assert!(!prompt.contains("README"));
    }

    #[test]
    fn user_prompt_prioritizes_1d_cut_definition_for_what_is_question() {
        let prompt = build_user_prompt("What is a 1D cut?", &[], &empty_snapshot())
            .expect("prompt should build");
        let hint_section = prompt
            .split("Codebase search results")
            .next()
            .expect("prompt should include hint section before codebase context");

        assert!(hint_section.contains("Source-derived concept hints for 1D cuts"));
        assert!(hint_section.contains("one-variable gate"));
        assert!(hint_section.contains("Polars boolean masks"));
        assert!(hint_section.contains("event observable"));
        assert!(!hint_section.contains("Source-derived workflow hints for 1D cuts"));
    }

    #[test]
    fn user_prompt_includes_2d_cut_workflow_hints() {
        let prompt = build_user_prompt("What is a 2D cut?", &[], &empty_snapshot())
            .expect("prompt should build");

        assert!(prompt.contains("Source-derived workflow hints for 2D cuts"));
        assert!(prompt.contains("Cut2D"));
        assert!(prompt.contains("right-click the plot"));
        assert!(prompt.contains("Double-click"));
        assert!(prompt.contains("`+2D` button loads a saved 2D cut JSON"));
    }

    #[test]
    fn user_prompt_includes_histogram_source_selection_hints() {
        let prompt = build_user_prompt(
            "How do I fill a 2D histogram from multiple columns without double counting?",
            &[],
            &empty_snapshot(),
        )
        .expect("prompt should build");

        assert!(prompt.contains("Source-derived workflow hints for histogram source selection"));
        assert!(prompt.contains("searchable multi-select pickers for both X and Y source columns"));
        assert!(prompt.contains("Column Groups"));
        assert!(prompt.contains("glob-style filters"));
        assert!(prompt.contains("skips self-pairs"));
        assert!(prompt.contains("suppresses mirrored duplicates"));
        assert!(prompt.contains(
            "2D cuts are only available when a 2D histogram has one unambiguous X/Y source pair"
        ));
    }

    #[test]
    fn user_prompt_includes_histogram_script_group_and_picker_hints() {
        let prompt = build_user_prompt(
            "How do column groups and grouped column creation work in Histogram Script?",
            &[],
            &empty_snapshot(),
        )
        .expect("prompt should build");

        assert!(prompt.contains("Source-derived workflow hints for Histogram Script setup"));
        assert!(prompt.contains("Column Groups"));
        assert!(prompt.contains("dragging the `↕` handle"));
        assert!(prompt.contains("glob-style filters"));
        assert!(prompt.contains("expands all combinations"));
        assert!(prompt.contains("`Output Name` field accepts `{}`"));
    }

    #[test]
    fn user_prompt_includes_gaussian_fit_workflow_hints() {
        let prompt = build_user_prompt("How do I fit Gaussian peaks?", &[], &empty_snapshot())
            .expect("prompt should build");

        assert!(prompt.contains("Source-derived workflow hints for Gaussian fitting"));
        assert!(prompt.contains("default background model is `None`"));
        assert!(prompt.contains("Press `B` to add a green background marker pair"));
        assert!(prompt.contains("Press `R` twice"));
        assert!(
            prompt.contains("Press `F` to perform the Gaussian fit with Spectrix's native Rust")
        );
        assert!(prompt.contains("`Temp` fit"));
        assert!(prompt.contains("`UUID = 0`"));
        assert!(prompt.contains("`Energy = -1`"));
    }

    #[test]
    fn user_prompt_includes_fit_workflow_hints_for_generic_fit_question() {
        let prompt = build_user_prompt("how do i do fits?", &[], &empty_snapshot())
            .expect("prompt should build");

        assert!(prompt.contains("Source-derived workflow hints for Gaussian fitting"));
        assert!(prompt.contains("Press `R` twice"));
        assert!(prompt.contains("Press `B`"));
        assert!(prompt.contains("Press `P` to add every peak marker manually"));
        assert!(prompt.contains("Editing one guess holds the other displayed variables fixed"));
        assert!(prompt.contains("Press `F`"));
        assert!(prompt.contains("Press `S`"));
    }

    #[test]
    fn user_prompt_includes_calibration_workflow_hints() {
        let prompt = build_user_prompt(
            "How do I calibrate a 1D histogram and make a calibrated column?",
            &[],
            &empty_snapshot(),
        )
        .expect("prompt should build");

        assert!(prompt.contains("Source-derived workflow hints for calibration"));
        assert!(prompt.contains("creating a new calibrated event-data column"));
        assert!(prompt.contains("assigned energy `-1`"));
        assert!(prompt.contains("a*x^2 + b*x + c"));
        assert!(prompt.contains("gain matching or shifting columns"));
        assert!(prompt.contains("two main paths"));
        assert!(prompt.contains("dedicated `Calibration` section"));
        assert!(prompt.contains("Import [.csv]"));
        assert!(prompt.contains("Column Creation"));
        assert!(prompt.contains("EnergyCalibrated"));
    }

    #[test]
    fn user_prompt_includes_calibration_workflow_hints_for_gain_matching_question() {
        let prompt = build_user_prompt(
            "How do I gain match or shift columns in Spectrix?",
            &[],
            &empty_snapshot(),
        )
        .expect("prompt should build");

        assert!(prompt.contains("Source-derived workflow hints for calibration"));
        assert!(prompt.contains("gain matching or shifting columns"));
        assert!(prompt.contains("dedicated `Calibration` section"));
    }

    #[test]
    fn user_prompt_includes_2d_projection_workflow_hints() {
        let prompt = build_user_prompt(
            "How do projections on 2D histograms work?",
            &[],
            &empty_snapshot(),
        )
        .expect("prompt should build");

        assert!(prompt.contains("Source-derived workflow hints for 2D projections"));
        assert!(prompt.contains("Press `X` to toggle an X projection"));
        assert!(prompt.contains("Press `Y` to toggle a Y projection"));
        assert!(prompt.contains("right-click the 2D plot"));
        assert!(prompt.contains("normal 1D histogram windows"));
    }

    #[test]
    fn user_prompt_can_include_multiple_workflow_hint_sections() {
        let prompt = build_user_prompt(
            "How do I fit Gaussian peaks, calibrate them, and make 2D projections?",
            &[],
            &empty_snapshot(),
        )
        .expect("prompt should build");

        assert!(prompt.contains("Source-derived workflow hints for Gaussian fitting"));
        assert!(prompt.contains("Source-derived workflow hints for calibration"));
        assert!(prompt.contains("Source-derived workflow hints for 2D projections"));
    }
}
