use crate::histoer::histogrammer::Histogrammer;

use polars::prelude::*;
use std::f64::consts::PI;

#[rustfmt::skip]
#[allow(clippy::all)]
pub fn manual_add_histograms(h: &mut Histogrammer, lf: LazyFrame) {

    // let lf = lf.with_columns(vec![
    //     (lit(0.5395)*col("PIPS1000Energy") + lit(2.5229)).alias("PIPS1000EnergyCalibrated")
    // ]);

    // h.add_fill_hist1d("PIPS1000", &lf, "PIPS1000Energy", 2000, (0.0, 2000.0), None);
    // h.add_fill_hist1d("PIPS1000_EnergyCalibrated", &lf, "PIPS1000EnergyCalibrated", 1200, (0.0, 1200.0), None);



    // /* 
    // Declare all the lazyframes that will be used
    let lf = lf.with_columns(vec![
        (col("DelayFrontRightEnergy") + col("DelayFrontLeftEnergy") / lit(2.0)).alias("DelayFrontAverageEnergy"),
        (col("DelayBackRightEnergy") + col("DelayBackLeftEnergy") / lit(2.0)).alias("DelayBackAverageEnergy"),
        (col("DelayFrontLeftTime") - col("AnodeFrontTime")).alias("DelayFrontLeftTime_AnodeFrontTime"),
        (col("DelayFrontRightTime") - col("AnodeFrontTime")).alias("DelayFrontRightTime_AnodeFrontTime"),
        (col("DelayBackLeftTime") - col("AnodeFrontTime")).alias("DelayBackLeftTime_AnodeFrontTime"),
        (col("DelayBackRightTime") - col("AnodeFrontTime")).alias("DelayBackRightTime_AnodeFrontTime"),
        (col("DelayFrontLeftTime") - col("AnodeBackTime")).alias("DelayFrontLeftTime_AnodeBackTime"),
        (col("DelayFrontRightTime") - col("AnodeBackTime")).alias("DelayFrontRightTime_AnodeBackTime"),
        (col("DelayBackLeftTime") - col("AnodeBackTime")).alias("DelayBackLeftTime_AnodeBackTime"),
        (col("DelayBackRightTime") - col("AnodeBackTime")).alias("DelayBackRightTime_AnodeBackTime"),
        (col("AnodeFrontTime") - col("AnodeBackTime")).alias("AnodeFrontTime_AnodeBackTime"),
        (col("AnodeBackTime") - col("AnodeFrontTime")).alias("AnodeBackTime_AnodeFrontTime"),
        (col("AnodeFrontTime") - col("ScintLeftTime")).alias("AnodeFrontTime_ScintLeftTime"),
        (col("AnodeBackTime") - col("ScintLeftTime")).alias("AnodeBackTime_ScintLeftTime"),
        (col("DelayFrontLeftTime") - col("ScintLeftTime")).alias("DelayFrontLeftTime_ScintLeftTime"),
        (col("DelayFrontRightTime") - col("ScintLeftTime")).alias("DelayFrontRightTime_ScintLeftTime"),
        (col("DelayBackLeftTime") - col("ScintLeftTime")).alias("DelayBackLeftTime_ScintLeftTime"),
        (col("DelayBackRightTime") - col("ScintLeftTime")).alias("DelayBackRightTime_ScintLeftTime"),
        (col("ScintRightTime") - col("ScintLeftTime")).alias("ScintRightTime_ScintLeftTime"),
        (lit(-0.013139237615)*col("Xavg")*col("Xavg") + lit(-13.80004977)*col("Xavg") + lit(9790.048149635)).alias("XavgEnergyCalibrated")
    ]);

    let lf_bothplanes = lf.clone().filter(col("X1").neq(lit(-1e6))).filter(col("X2").neq(lit(-1e6)));
    let lf_only_x1_plane = lf.clone().filter(col("X1").neq(lit(-1e6))).filter(col("X2").eq(lit(-1e6)));
    let lf_only_x2_plane = lf.clone().filter(col("X2").neq(lit(-1e6))).filter(col("X1").eq(lit(-1e6)));


    //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

    // Focal Plane histograms

    let fp_grid = Some("Focal Plane"); // Creates a new grid container

    h.add_fill_hist1d("X1", &lf, "X1", 600, (-300.0, 300.0), fp_grid);
    h.add_fill_hist1d("X1: only1plane", &lf_only_x1_plane, "X1", 600, (-300.0, 300.0), fp_grid);
    h.add_fill_hist1d("X1: bothplanes", &lf_bothplanes, "X1", 600, (-300.0, 300.0), fp_grid);
    h.add_fill_hist1d("X2", &lf, "X2", 600, (-300.0, 300.0), fp_grid);
    h.add_fill_hist1d("X2: only1plane", &lf_only_x2_plane, "X2", 600, (-300.0, 300.0), fp_grid);
    h.add_fill_hist1d("X2: bothplanes", &lf_bothplanes, "X2", 600, (-300.0, 300.0), fp_grid);
    h.add_fill_hist2d("X2 v X1", &lf, "X1", "X2", (600, 600), ((-300.0, 300.0), (-300.0, 300.0)), fp_grid);
    h.add_fill_hist1d("Xavg: bothplanes", &lf_bothplanes, "Xavg", 600, (-300.0, 300.0), fp_grid);
    h.add_fill_hist2d("Theta v Xavg: bothplanes", &lf_bothplanes, "Xavg", "Theta", (600, 300), ((-300.0, 300.0), (0.0, PI)), fp_grid);
    // h.add_fill_hist1d("XavgEnergyCalibrated", &lf, "XavgEnergyCalibrated", 4096, (0.0, 16384.0), fp_grid);

    // // // ....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

    // Particle Identification histograms

    let pid_grid = Some("Particle Identification"); 

    h.add_fill_hist2d("AnodeBack v ScintLeft", &lf, "ScintLeftEnergy", "AnodeBackEnergy", (512, 512), ((0.0, 4096.0), (0.0, 4096.0)), pid_grid);
    h.add_fill_hist2d("AnodeFront v ScintLeft", &lf, "ScintLeftEnergy", "AnodeFrontEnergy", (512, 512), ((0.0, 4096.0), (0.0, 4096.0)), pid_grid);
    h.add_fill_hist2d("Cathode v ScintLeft", &lf, "ScintLeftEnergy", "CathodeEnergy", (512, 512), ((0.0, 4096.0), (0.0, 4096.0)), pid_grid);
    h.add_fill_hist2d("AnodeBack v ScintRight", &lf, "ScintRightEnergy", "AnodeBackEnergy", (512, 512), ((0.0, 4096.0), (0.0, 4096.0)), pid_grid);
    h.add_fill_hist2d("AnodeFront v ScintRight", &lf, "ScintRightEnergy", "AnodeFrontEnergy", (512, 512), ((0.0, 4096.0), (0.0, 4096.0)), pid_grid);
    h.add_fill_hist2d("Cathode v ScintRight", &lf, "ScintRightEnergy", "CathodeEnergy", (512, 512), ((0.0, 4096.0), (0.0, 4096.0)), pid_grid);
    
    // // //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

    // // Particle Identification vs Focal plane histograms

    let pid_v_fp_grid = Some("Particle Identification v Focal Plane");

    h.add_fill_hist2d("ScintLeft v X1", &lf, "X1", "ScintLeftEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("ScintLeft v X2", &lf, "X2", "ScintLeftEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("ScintLeft v Xavg", &lf, "Xavg", "ScintLeftEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("ScintRight v X1", &lf, "X1", "ScintRightEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("ScintRight v X2", &lf, "X2", "ScintRightEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("ScintRight v Xavg", &lf, "Xavg", "ScintRightEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("AnodeBack v X1", &lf, "X1", "AnodeBackEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("AnodeBack v X2", &lf, "X2", "AnodeBackEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("AnodeBack v Xavg", &lf, "Xavg", "AnodeBackEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("AnodeFront v X1", &lf, "X1", "AnodeFrontEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("AnodeFront v X2", &lf, "X2", "AnodeFrontEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("AnodeFront v Xavg", &lf, "Xavg", "AnodeFrontEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("Cathode v X1", &lf, "X1", "CathodeEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("Cathode v X2", &lf, "X2", "CathodeEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);
    h.add_fill_hist2d("Cathode v Xavg", &lf, "Xavg", "CathodeEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), pid_v_fp_grid);

    // // // //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

    // // Delay lines vs Focal plane histograms

    let de_v_fp_grid = Some("Delay Lines v Focal Plane");

    // h.add_fill_hist2d("DelayBackRight v X1", &lf, "X1", "DelayBackRightEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    // h.add_fill_hist2d("DelayBackLeft v X1", &lf, "X1", "DelayBackLeftEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    h.add_fill_hist2d("DelayFrontRight v X1", &lf, "X1", "DelayFrontRightEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    h.add_fill_hist2d("DelayFrontLeft v X1", &lf, "X1", "DelayFrontLeftEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    h.add_fill_hist2d("DelayBackRight v X2", &lf, "X2", "DelayBackRightEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    h.add_fill_hist2d("DelayBackLeft v X2", &lf, "X2", "DelayBackLeftEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    // h.add_fill_hist2d("DelayFrontRight v X2", &lf, "X2", "DelayFrontRightEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    // h.add_fill_hist2d("DelayFrontLeft v X2", &lf, "X2", "DelayFrontLeftEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    h.add_fill_hist2d("DelayBackRight v Xavg", &lf, "Xavg", "DelayBackRightEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    h.add_fill_hist2d("DelayBackLeft v Xavg", &lf, "Xavg", "DelayBackLeftEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    h.add_fill_hist2d("DelayFrontRight v Xavg", &lf, "Xavg", "DelayFrontRightEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    h.add_fill_hist2d("DelayFrontLeft v Xavg", &lf, "Xavg", "DelayFrontLeftEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    // h.add_fill_hist2d("DelayFrontAverage v X1", &lf, "X1", "DelayFrontAverageEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    // h.add_fill_hist2d("DelayBackAverage v X1", &lf, "X1", "DelayBackAverageEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    // h.add_fill_hist2d("DelayFrontAverage v X2", &lf, "X2", "DelayFrontAverageEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    // h.add_fill_hist2d("DelayBackAverage v X2", &lf, "X2", "DelayBackAverageEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    // h.add_fill_hist2d("DelayFrontAverage v Xavg", &lf, "Xavg", "DelayFrontAverageEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    // h.add_fill_hist2d("DelayBackAverage v Xavg", &lf, "Xavg", "DelayBackAverageEnergy", (600, 512), ((-300.0, 300.0), (0.0, 4096.0)), de_v_fp_grid);
    
    //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
    /* 
    // Delay timing relative to anodes histograms

    h.add_fill_hist1d("DelayFrontLeftTime-AnodeFrontTime: bothplanes", &lf_bothplanes, "DelayFrontLeftTime_AnodeFrontTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayFrontRightTime-AnodeFrontTime: bothplanes", &lf_bothplanes, "DelayFrontRightTime_AnodeFrontTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayBackLeftTime-AnodeBackTime: bothplanes", &lf_bothplanes, "DelayBackLeftTime_AnodeBackTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayBackRightTime-AnodeBackTime: bothplanes", &lf_bothplanes, "DelayBackRightTime_AnodeBackTime", 8000, (-4000.0, 4000.0));

    h.add_fill_hist1d("DelayFrontLeftTime-AnodeFrontTime: onlyX1", &lf_only_x1_plane, "DelayFrontLeftTime_AnodeFrontTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayFrontRightTime-AnodeFrontTime: onlyX1", &lf_only_x1_plane, "DelayFrontRightTime_AnodeFrontTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayBackLeftTime-AnodeFrontTime: onlyX1", &lf_only_x1_plane, "DelayBackLeftTime_AnodeFrontTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayBackRightTime-AnodeFrontTime: onlyX1", &lf_only_x1_plane, "DelayBackRightTime_AnodeFrontTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayFrontLeftTime-AnodeBackTime: onlyX1", &lf_only_x1_plane, "DelayFrontLeftTime_AnodeBackTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayFrontRightTime-AnodeBackTime: onlyX1", &lf_only_x1_plane, "DelayFrontRightTime_AnodeBackTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayBackLeftTime-AnodeBackTime: onlyX1", &lf_only_x1_plane, "DelayBackLeftTime_AnodeBackTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayBackRightTime-AnodeBackTime: onlyX1", &lf_only_x1_plane, "DelayBackRightTime_AnodeBackTime", 8000, (-4000.0, 4000.0));

    h.add_fill_hist1d("DelayFrontLeftTime-AnodeFrontTime: onlyX2", &lf_only_x2_plane, "DelayFrontLeftTime_AnodeFrontTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayFrontRightTime-AnodeFrontTime: onlyX2", &lf_only_x2_plane, "DelayFrontRightTime_AnodeFrontTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayBackLeftTime-AnodeFrontTime: onlyX2", &lf_only_x2_plane, "DelayBackLeftTime_AnodeFrontTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayBackRightTime-AnodeFrontTime: onlyX2", &lf_only_x2_plane, "DelayBackRightTime_AnodeFrontTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayFrontLeftTime-AnodeBackTime: onlyX2", &lf_only_x2_plane, "DelayFrontLeftTime_AnodeBackTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayFrontRightTime-AnodeBackTime: onlyX2", &lf_only_x2_plane, "DelayFrontRightTime_AnodeBackTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayBackLeftTime-AnodeBackTime: onlyX2", &lf_only_x2_plane, "DelayBackLeftTime_AnodeBackTime", 8000, (-4000.0, 4000.0));
    h.add_fill_hist1d("DelayBackRightTime-AnodeBackTime: onlyX2", &lf_only_x2_plane, "DelayBackRightTime_AnodeBackTime", 8000, (-4000.0, 4000.0));

    let lf_time_rel_backanode = lf.clone().filter(col("AnodeBackTime").neq(lit(-1e6))).filter(col("ScintLeftTime").neq(lit(-1e6)));
    h.add_fill_hist1d("AnodeFrontTime-AnodeBackTime", &lf_time_rel_backanode, "AnodeFrontTime_AnodeBackTime", 1000, (-3000.0, 3000.0));
    h.add_fill_hist1d("AnodeBackTime-AnodeFrontTime", &lf_time_rel_backanode, "AnodeBackTime_AnodeFrontTime", 1000, (-3000.0, 3000.0));
    h.add_fill_hist1d("AnodeFrontTime-ScintLeftTime", &lf_time_rel_backanode, "AnodeFrontTime_ScintLeftTime", 1000, (-3000.0, 3000.0));
    h.add_fill_hist1d("AnodeBackTime-ScintLeftTime", &lf_time_rel_backanode, "AnodeBackTime_ScintLeftTime", 1000, (-3000.0, 3000.0));
    h.add_fill_hist1d("DelayFrontLeftTime-ScintLeftTime", &lf_time_rel_backanode, "DelayFrontLeftTime_ScintLeftTime", 1000, (-3000.0, 3000.0));
    h.add_fill_hist1d("DelayFrontRightTime-ScintLeftTime", &lf_time_rel_backanode, "DelayFrontRightTime_ScintLeftTime", 1000, (-3000.0, 3000.0));
    h.add_fill_hist1d("DelayBackLeftTime-ScintLeftTime", &lf_time_rel_backanode, "DelayBackLeftTime_ScintLeftTime", 1000, (-3000.0, 3000.0));
    h.add_fill_hist1d("DelayBackRightTime-ScintLeftTime", &lf_time_rel_backanode, "DelayBackRightTime_ScintLeftTime", 1000, (-3000.0, 3000.0));
    h.add_fill_hist1d("ScintRightTime-ScintLeftTime", &lf_time_rel_backanode, "ScintRightTime_ScintLeftTime", 1000, (-3000.0, 3000.0));
    h.add_fill_hist2d("ScintTimeDif v Xavg", &lf_time_rel_backanode, "Xavg", "ScintRightTime_ScintLeftTime", (600, 12800), ((-300.0, 300.0), (-3200.0, 3200.0)));

    let delay_timing_panes_names = vec![
        "DelayFrontLeftTime-AnodeFrontTime: bothplanes", "DelayFrontRightTime-AnodeFrontTime: bothplanes",
        "DelayBackLeftTime-AnodeBackTime: bothplanes", "DelayBackRightTime-AnodeBackTime: bothplanes",
        "DelayFrontLeftTime-AnodeFrontTime: onlyX1", "DelayFrontRightTime-AnodeFrontTime: onlyX1",
        "DelayBackLeftTime-AnodeFrontTime: onlyX1", "DelayBackRightTime-AnodeFrontTime: onlyX1",
        "DelayFrontLeftTime-AnodeBackTime: onlyX1", "DelayFrontRightTime-AnodeBackTime: onlyX1",
        "DelayBackLeftTime-AnodeBackTime: onlyX1", "DelayBackRightTime-AnodeBackTime: onlyX1",
        "DelayFrontLeftTime-AnodeFrontTime: onlyX2", "DelayFrontRightTime-AnodeFrontTime: onlyX2",
        "DelayBackLeftTime-AnodeFrontTime: onlyX2", "DelayBackRightTime-AnodeFrontTime: onlyX2",
        "DelayFrontLeftTime-AnodeBackTime: onlyX2", "DelayFrontRightTime-AnodeBackTime: onlyX2",
        "DelayBackLeftTime-AnodeBackTime: onlyX2", "DelayBackRightTime-AnodeBackTime: onlyX2",
        "AnodeFrontTime-AnodeBackTime", "AnodeBackTime-AnodeFrontTime",
        "AnodeFrontTime-ScintLeftTime", "AnodeBackTime-ScintLeftTime",
        "DelayFrontLeftTime-ScintLeftTime", "DelayFrontRightTime-ScintLeftTime",
        "DelayBackLeftTime-ScintLeftTime", "DelayBackRightTime-ScintLeftTime",
        "ScintRightTime-ScintLeftTime", "ScintTimeDif v Xavg"
    ];

    let delay_timing_panes = h.get_panes(delay_timing_panes_names);

    h.tabs.insert("Delay Timing".to_string(), delay_timing_panes);
    */
    //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

    /*
    // CeBrA plots
    // declare the gain matched energy columns, and time to scint left columns

    let time_shits = vec![
        1158.5,
        1162.5,
        1160.5,
        1160.5,
        1135.5,
        1130.5,
        1129.5,
        1128.5,
        1136.5,
    ];

    let gain_match_values = vec![
        (-2.72156900878375e-19, 1.0, -5.6843418860808e-14),
        (-4.97165415075812e-06, 1.14338957961188, -0.445011719623494),
        (-2.35730449363139e-05, 0.996173496729512, 1.7443869915702),
        (-1.32712886414671e-05, 0.999818365276875, 0.941072453624997),
        (-2.76775763630926e-05, 0.860232802761158, -0.459548374380745),
        (-2.22330292983274e-05, 1.2289729811605, 0.283037829606826),
        (-4.12687972463716e-07, 0.701053103610418, 0.211631359048795),
        (-3.57953438680052e-05, 1.29207015301957, 1.07805724372673),
        (-3.03544968865592e-05, 0.957884129953653, -1.25796690476597),
    ];

    let time_cut_width = 6.0;

    let mut det_lfs = Vec::new();
    let mut det_time_lfs = Vec::new();
    let mut det_time_cut_lfs = Vec::new();
    for i in 0..9 {
        let det_lf = lf.clone()
            .filter(col(&format!("Cebra{}Energy", i)).neq(lit(-1e6)))
            .with_column(
                (lit(gain_match_values[i].0)*col(&format!("Cebra{}Energy", i))*col(&format!("Cebra{}Energy", i)) 
                    + lit(gain_match_values[i].1)*col(&format!("Cebra{}Energy", i)) 
                    + lit(gain_match_values[i].2)
                ).alias(&format!("Cebra{}EnergyGM", i)))
            .with_column((col(&format!("Cebra{}Time", i)) - col("ScintLeftTime")).alias(&format!("Cebra{}Time_ScintLeftTime", i)));

        let det_time_lf = det_lf.clone()
            .filter(col(&format!("Cebra{}Time", i)).neq(lit(-1e6)))
            .filter(col("ScintLeftTime").neq(lit(-1e6)))
            .filter(col("AnodeBackTime").neq(lit(-1e6)))
            .with_column((col(&format!("Cebra{}Time_ScintLeftTime", i)) + lit(time_shits[i])).alias(&format!("Cebra{}Time_ScintLeftTime_Shifted", i)));
    
        let det_time_cut_lf = det_time_lf.clone()
            .filter(col(&format!("Cebra{}Time_ScintLeftTime_Shifted", i)).gt(-time_cut_width))
            .filter(col(&format!("Cebra{}Time_ScintLeftTime_Shifted", i)).lt(time_cut_width))
            .with_column(( lit(2.598957532)*col(&format!("Cebra{}EnergyGM", i)) + lit(-12.36847047833) ).alias(&format!("Cebra{}EnergyECal", i)));

        det_lfs.push(det_lf);
        det_time_lfs.push(det_time_lf);
        det_time_cut_lfs.push(det_time_cut_lf);
    }

    h.add_hist1d("CeBrA Gain Matched", 512, (0.0, 4096.0));
    h.add_hist1d("CeBrA Time to Scint Shifted with TCut", 100, (-50.0, 50.0));
    h.add_hist2d("CeBrA Gain Matched vs Xavg with TCut", (600,512), ((-300.0, 300.0),(0.0, 4096.0)));
    h.add_hist2d("CeBrA vs Xavg: Energy Calibrated", (4096,512), ((0.0, 16384.0),(0.0, 8192.0)));

    
    for i in 0..9 {
        // Raw CeBrA Histograms
        h.add_fill_hist1d(&format!("Cebra{}Energy", i), &det_lfs[i], &format!("Cebra{}Energy", i), 512, (0.0, 4096.0));
        h.add_fill_hist2d(&format!("Cebra{}Energy v Xavg", i), &det_lfs[i], "Xavg", &format!("Cebra{}Energy", i), (600, 512), ((-300.0, 300.0), (0.0, 4096.0)));
        h.add_fill_hist1d(&format!("Cebra{}Time-ScintLeftTime", i), &det_time_lfs[i], &format!("Cebra{}Time_ScintLeftTime", i), 3200, (-1600.0, 1600.0));
        h.add_fill_hist1d(&format!("Cebra{}Time-ScintLeftTime Shifted", i), &det_time_lfs[i], &format!("Cebra{}Time_ScintLeftTime_Shifted", i), 3200, (-1600.0, 1600.0));

        // Gain Matched Histograms
        h.add_fill_hist1d(&format!("Cebra{}EnergyGM", i), &det_lfs[i], &format!("Cebra{}EnergyGM", i), 512, (0.0, 4096.0));
        h.add_fill_hist2d(&format!("Cebra{}EnergyGainMatched v Xavg", i), &det_lfs[i], "Xavg", &format!("Cebra{}EnergyGM", i), (600, 512), ((-300.0, 300.0), (0.0, 4096.0)));

        // Time Cut Histograms
        h.add_fill_hist1d(&format!("Cebra{}Time-ScintLeftTime Shifted Time Cut", i), &det_time_cut_lfs[i], &format!("Cebra{}Time_ScintLeftTime_Shifted", i), 100, (-50.0, 50.0));
        h.add_fill_hist1d(&format!("Cebra{}Energy Time Cut", i), &det_time_cut_lfs[i], &format!("Cebra{}Energy", i), 512, (0.0, 4096.0));
        h.add_fill_hist2d(&format!("Cebra{}Energy v Xavg Time Cut", i), &det_time_cut_lfs[i], "Xavg", &format!("Cebra{}Energy", i), (600, 512), ((-300.0, 300.0), (0.0, 4096.0)));

        h.add_fill_hist2d(&format!("Cebra{}EnergyGM v Xavg Time Cut", i), &det_time_cut_lfs[i], "Xavg", &format!("Cebra{}EnergyGM", i), (600, 512), ((-300.0, 300.0), (0.0, 4096.0)));
        
        let cebra_det_panes_name_strings: Vec<String> = vec![
            format!("Cebra{}Energy", i), 
            format!("Cebra{}Energy v Xavg", i), 
            format!("Cebra{}Time-ScintLeftTime", i), 
            format!("Cebra{}Time-ScintLeftTime Shifted", i),
            format!("Cebra{}EnergyGM", i), 
            format!("Cebra{}EnergyGainMatched v Xavg", i),
            format!("Cebra{}Time-ScintLeftTime Shifted Time Cut", i), 
            format!("Cebra{}Energy Time Cut", i), 
            format!("Cebra{}Energy v Xavg Time Cut", i),
            format!("Cebra{}EnergyGM v Xavg Time Cut", i)
        ];

        let cebra_det_panes_name: Vec<&str> = cebra_det_panes_name_strings.iter().map(|s| s.as_str()).collect();

        let cebra_det_panes = h.get_panes(cebra_det_panes_name);

        h.tabs.insert(format!("CeBr3 Detector {}", i).to_string(), cebra_det_panes);

        // fill the combined histograms
        h.fill_hist1d(&format!("CeBrA Gain Matched"), &det_lfs[i], &format!("Cebra{}EnergyGM", i));
        h.fill_hist1d(&format!("CeBrA Time to Scint Shifted with TCut"), &det_time_cut_lfs[i], &format!("Cebra{}Time_ScintLeftTime_Shifted", i));
        h.fill_hist2d(&format!("CeBrA Gain Matched vs Xavg with TCut"), &det_time_cut_lfs[i], "Xavg", &format!("Cebra{}EnergyGM", i));
        h.fill_hist2d(&format!("CeBrA vs Xavg: Energy Calibrated"), &det_time_cut_lfs[i], "XavgEnergyCalibrated", &format!("Cebra{}EnergyECal", i));
    }

    //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
    let cebra_panes_names = vec![
        "CeBrA Gain Matched", "CeBrA Time to Scint Shifted with TCut", 
        "CeBrA Gain Matched vs Xavg with TCut", "CeBrA vs Xavg: Energy Calibrated"
    ];

    let cebra_panes = h.get_panes(cebra_panes_names);
    h.tabs.insert("CeBrA".to_string(), cebra_panes);

    */

    
    // */
}
