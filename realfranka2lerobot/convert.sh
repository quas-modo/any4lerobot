export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python realfranka_h5.py \
    --src-paths /oss/shared/tr-eval/lianghaotian/franka_real_data/stack_cup \
    --output-path /cpfs04/user/lianghaotian/code_xy/any4lerobot/realfranka2lerobot/data \
    --executor local \
    --tasks-per-job 3 \
    --workers 10 \
    --resume-from-save /cpfs04/user/lianghaotian/code_xy/any4lerobot/logs/resume_from_save \
    --resume-from-aggregate /cpfs04/user/lianghaotian/code_xy/any4lerobot/logs/resume_from_aggregate  \
    --repo-id quasdo1002/realdata_stack_cup \

 # --output-path /oss/shared/tr-eval/lianghaotian/franka_real_data/franka_real_data_lerobot/ \