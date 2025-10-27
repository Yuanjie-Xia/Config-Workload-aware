# Config-Workload-aware

This is a supplementary of paper.

Main stream:

base_model_method_*.py: Running script of results of RQ1.

find_stable_case*.py: Running script of PerfFuzz-PM.

analyze_three_type*.py: Analyzing three performance conditions based on fuzz method results.

results/: results of evaluations

figures*/: estimation results of three performance conditions

Since we test four system, the default one is the TeaStore architecture with ffmpeg. "_mix" represents TeaStore architecture with lrzip and ffmpeg. "_train" represents TrainTicket with ffmpeg.
"_train_mix" represents TrainTicket with ffmpeg and lrzip.