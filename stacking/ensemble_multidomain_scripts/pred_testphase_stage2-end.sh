
set -eu

train_datatrack=phase1-main
pred_datatrack=testphase-main
feat_type=main17-weak48s-indomain-opt

# train_datatrack=phase1-ood
# pred_datatrack=testphase-ood
# feat_type=ood6-weak48s-opt

poetry  run python -u collect_stage1_testphase_result.py ${pred_datatrack} ${feat_type}

for method in exactgp ridge linear_svr kernel_svr rf lightgbm; do
for i_cv in 0 1 2 3 4; do
    echo "${method}, ${train_datatrack}, ${feat_type}, ${i_cv}, ${pred_datatrack}"
    poetry run python -u pred_testphase_stage2.py ${method} ${train_datatrack} ${feat_type} ${i_cv} ${pred_datatrack}
done
done

echo "Collect stage2 data."
poetry  run python -u collect_stage2_testphase_result.py ${pred_datatrack} ${feat_type}

for method in ridge; do
for i_cv in 0 1 2 3 4; do
    echo "Run stage3: ${method}, ${train_datatrack}, ${feat_type}, ${i_cv}, ${pred_datatrack}"
    poetry run python -u pred_testphase_stage3.py ${method} ${train_datatrack} ${feat_type} ${i_cv} ${pred_datatrack}
done
done

echo "Calculate result."

poetry run python -u calc_testphase_result.py ${pred_datatrack} ${feat_type}

echo "done"
