SEED=123
MODEL_NAME='baseline_v0_training_free'
BS=4 #! 4 for 'total' and 'open' test data to ensure all test data are evaluated
test_data_type='close' # 32 for 'close' test data
test_strategy_type='v01' # 'v0' or 'v01', default: v01, best

python baseline_v0_training_free.py \
--lr 0.001 \
--seed ${SEED} \
--model ${MODEL_NAME} \
--snapshot_pref "./ExpResults/${MODEL_NAME}/test_mode_seed${SEED}_bs${BS}_dataType_${test_data_type}_strategy_${test_strategy_type}" \
--train_batch_size ${BS} \
--test_batch_size ${BS} \
--evaluate \
--print_iter_freq 100 \
--test_data_type ${test_data_type} \
--test_strategy_type ${test_strategy_type}