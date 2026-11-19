# Validation of the cleaned trace bundles

## cross-dataset comparison

| dataset | jobs_rows | gpu_job_rows | gpu_job_pct | machines | gpu_devices | gpu_labels | snapshots | placements_rows | placement_devices_rows | users | accounts | median_wait_s | median_run_s | jobs_label_distinct_values | jobs_label_composite_pct | placement_labels_not_in_job_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ali_v2020 | 1,055,501 | 1,042,992 | 98.810 | 1,903 | 6,829 | 6 | 69 | 7,190,867.000 | 3,033,232.000 | 1,433 | 317,988 | 0.000 | 634.000 | 35 | 8.960 | 0.000 |
| philly | 117,325 | 112,018 | 95.480 | 552 | 3,918 | 2 | 124 | 179,828.000 | 343,082.000 | 325 | 15 | 131.000 | 1,269.000 | 3 | 5.240 | 0.000 |
| eagle | 11,030,377 | 122,318 | 1.110 | 0 | 0 | 0 | 1,528 | -- | -- | 936 | 594 | 5,427.000 | 336.000 | 0 | 0.000 | -- |

## per-dataset detail

### ali_v2020

- `jobs_rows`: 1055501
- `jobs_cols`: 46
- `pk_duplicate_rows`: 0
- `native_id_missing_rows`: 0
- `placements_rows`: 7190867
- `placements:job_gid_orphan_rows`: 0
- `placements:machine_gid_null_key_rows`: 26509
- `placements:machine_gid_orphan_rows`: 0
- `placement_devices_rows`: 3033232
- `placement_devices:job_gid_null_key_rows`: 8713
- `placement_devices:job_gid_orphan_rows`: 0
- `placement_devices:machine_gid_orphan_rows`: 0
- `placement_devices:device_slot_unresolved`: 0
- `placement_devices:device_gid_orphan_rows`: 0
- `requests_rows`: 1261050
- `requests:job_gid_orphan_rows`: 0
- `devices_orphan_machines`: 0
- `placements_labels`: ['CPU-ONLY', 'NVIDIA-LEGACY-MISC', 'NVIDIA-P100', 'NVIDIA-T4', 'NVIDIA-V100', 'NVIDIA-V100-32G', 'UNRESOLVED']
- `placement_devices_labels`: ['NVIDIA-LEGACY-MISC', 'NVIDIA-P100', 'NVIDIA-T4', 'NVIDIA-V100', 'NVIDIA-V100-32G']
- `devices_labels`: ['NVIDIA-LEGACY-MISC', 'NVIDIA-P100', 'NVIDIA-T4', 'NVIDIA-V100', 'NVIDIA-V100-32G', 'UNRESOLVED']
- `t_submit_s:min`: 0.0
- `t_submit_s:max`: 5956762.0
- `t_submit_s:nan_pct`: 0.347
- `t_start_s:min`: 0.0
- `t_start_s:max`: 5956762.0
- `t_start_s:nan_pct`: 0.054
- `t_end_s:min`: 237453.0
- `t_end_s:max`: 5956873.0
- `t_end_s:nan_pct`: 28.083
- `snapshots`: 69
- `snap_range`: [0.0, 68.0]
- `negative_wait_s`: 0
- `median_wait_s`: 0.0
- `negative_run_s`: 0
- `median_run_s`: 634.0
- `gpu_job_rows`: 1042992
- `gpu_job_pct`: 98.81
- `users`: 1433
- `accounts`: 317988
- `machines`: 1903
- `gpu_devices`: 6829
- `gpu_labels`: 6
- `missing_core_job_columns`: []
- `dataset_specific_job_columns`: ['request_id', 'framework', 'n_tasks', 'n_gpu_machines', 'gpu_type_requested_raw']
- `jobs_label_distinct_values`: 35
- `jobs_label_composite_rows`: 94518
- `jobs_label_composite_pct`: 8.96
- `placement_labels_not_in_job_label`: 0
- `jobs_label_elements_outside_vocabulary`: []
- `manifest_absent_tables`: []

### philly

- `jobs_rows`: 117325
- `jobs_cols`: 42
- `pk_duplicate_rows`: 0
- `native_id_missing_rows`: 0
- `placements_rows`: 179828
- `placements:job_gid_orphan_rows`: 0
- `placements:machine_gid_orphan_rows`: 0
- `placement_devices_rows`: 343082
- `placement_devices:job_gid_orphan_rows`: 0
- `placement_devices:machine_gid_orphan_rows`: 0
- `placement_devices:device_slot_unresolved`: 0
- `placement_devices:device_gid_orphan_rows`: 0
- `devices_orphan_machines`: 0
- `placements_labels`: ['PHILLY-TIER-2x12', 'PHILLY-TIER-8x24']
- `placement_devices_labels`: ['PHILLY-TIER-2x12', 'PHILLY-TIER-8x24']
- `devices_labels`: ['PHILLY-TIER-2x12', 'PHILLY-TIER-8x24']
- `t_submit_s:min`: 0.0
- `t_submit_s:max`: 11833150.0
- `t_submit_s:nan_pct`: 0.01
- `t_start_s:min`: 2223721.0
- `t_start_s:max`: 11833157.0
- `t_start_s:nan_pct`: 4.534
- `t_end_s:min`: 3317063.0
- `t_end_s:max`: 11833221.0
- `t_end_s:nan_pct`: 6.3
- `snapshots`: 124
- `snap_range`: [0.0, 136.0]
- `negative_wait_s`: 0
- `median_wait_s`: 131.0
- `negative_run_s`: 0
- `median_run_s`: 1269.0
- `gpu_job_rows`: 112018
- `gpu_job_pct`: 95.48
- `users`: 325
- `accounts`: 15
- `machines`: 552
- `gpu_devices`: 3918
- `gpu_labels`: 2
- `missing_core_job_columns`: []
- `dataset_specific_job_columns`: ['n_attempts']
- `jobs_label_distinct_values`: 3
- `jobs_label_composite_rows`: 5868
- `jobs_label_composite_pct`: 5.24
- `placement_labels_not_in_job_label`: 0
- `jobs_label_elements_outside_vocabulary`: []
- `manifest_absent_tables`: ['requests']

### eagle

- `jobs_rows`: 11030377
- `jobs_cols`: 46
- `pk_duplicate_rows`: 0
- `native_id_missing_rows`: 0
- `t_submit_s:min`: 25200.0
- `t_submit_s:max`: 134850459.0
- `t_submit_s:nan_pct`: 0.001
- `t_start_s:min`: 25201.0
- `t_start_s:max`: 134850989.0
- `t_start_s:nan_pct`: 0.165
- `t_end_s:min`: 25278.0
- `t_end_s:max`: 134851076.0
- `t_end_s:nan_pct`: 0.176
- `snapshots`: 1528
- `snap_range`: [0.0, 1560.0]
- `negative_wait_s`: 1
- `median_wait_s`: 5427.0
- `negative_run_s`: 0
- `median_run_s`: 336.0
- `gpu_job_rows`: 122318
- `gpu_job_pct`: 1.11
- `users`: 936
- `accounts`: 594
- `machines`: 0
- `gpu_devices`: 0
- `gpu_labels`: 0
- `missing_core_job_columns`: []
- `dataset_specific_job_columns`: ['n_nodes_req', 'wallclock_limit_s', 'wallclock_overrun', 'work_dir', 'submit_line']
- `jobs_label_distinct_values`: 0
- `jobs_label_composite_rows`: 0
- `jobs_label_composite_pct`: 0.0
- `jobs_label_elements_outside_vocabulary`: []
- `manifest_absent_tables`: ['devices', 'gpu_types', 'nodes', 'placement_devices', 'placements', 'requests']

## failures

none -- H1..H8 passed for every dataset checked
