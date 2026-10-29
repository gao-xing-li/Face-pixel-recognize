# C independent test evidence

Source A: 1389a77158ab484dbdbbbb4e4e2f4d1c46a9625b. Source B: 19e1d7409991c5896dfca9a6ef03694d2276a152.

independent_test.json records immutable configurations, source hashes, environment versions and all four evaluation summaries. test_final/test_baseline contain per-image counts and exact configuration snapshots. validation_final/validation_baseline record the independent replay.

case_analysis.json describes already fixed intermediate masks, without selecting new parameters. PNG case_path fields refer to the complete run directory included in the downloadable results archive; PNGs and original images are not duplicated in Git.

Final test F1=0.7069326974722259, IoU=0.5467099014028634. Baseline test F1=0.6266214969531303, IoU=0.45626278230142453. Inference used fixed B code; evaluation used independent C ground truth.

visual_assessment/ contains supplementary post-hoc boundary, image-level uncertainty and brightness-group descriptions of the same frozen predictions. These do not replace the original test or create a new independent benchmark. Figure PNG/SVG files are in the downloadable visualization package and can be regenerated with face-pixel-visualize.
