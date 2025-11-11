# Config-Workload-aware

This is a supplement to the paper.

Mainstream:

base_model_method_*.py: Running script of results of our approach.

find_stable_case*.py: Running script of our search-based method.

analyze_three_type*.py: Analyzing three performance conditions based on search-based method results.

results/: results of evaluations

figures*/: estimation results of three performance conditions

Since we test four systems, the default one is the TeaStore architecture with ffmpeg. "_mix" represents TeaStore architecture with lrzip and ffmpeg. "_train" represents TrainTicket with ffmpeg.
"_train_mix" represents TrainTicket with ffmpeg and lrzip.
