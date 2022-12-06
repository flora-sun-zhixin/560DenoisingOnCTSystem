filename = "/Users/flora/Docs/MathImgSci/results/pred_Very_High_Noise.hdf5";
data = h5read(filename, "/predict");
IN = reshape(data(:, :, 1:100), [], 100);
IS = reshape(data(:, :, 101:200), [], 100);
[AUC] = performance_evaluation(IS,IN);
