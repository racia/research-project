baseline_reasoning_v1_t_2 = [
    "basic-baseline/test/reasoning/v1/unjoined/task_1_full_task_2_s_1_27",
    "basic-baseline/test/reasoning/v1/unjoined/task_2_s_27_100",
]
baseline_reasoning_v1_t_5 = [
    "basic-baseline/test/reasoning/v1/unjoined/task_3_full_task_5_1_94",
    "basic-baseline/test/reasoning/v1/unjoined/task_5_94_100_task_14_full",
]
baseline_reasoning_v1_t_8 = [
    "basic-baseline/test/reasoning/v1/unjoined/task_6_7_full_task_8_s_1_77",
    "basic-baseline/test/reasoning/v1/unjoined/task_8_s_77_100",
]
baseline_reasoning_v1_t_20 = [
    "basic-baseline/test/reasoning/v1/unjoined/task_17_18_full_task_20_s_1_15",
    "basic-baseline/test/reasoning/v1/unjoined/task_20_s_9_27",
    "basic-baseline/test/reasoning/v1/unjoined/task_20_s_27_48",
    "basic-baseline/test/reasoning/v1/unjoined/task_20_s_48_93",
]
baseline_reasoning_v1 = [
    "basic-baseline/test/reasoning/v1/task_15_16",
    *[
        f"basic-baseline/test/reasoning/v1/task_{i}"
        for i in range(1, 21)
        if i not in [15, 16]
    ],
]

baseline_reasoning_v2_t_14 = [
    "basic-baseline/test/reasoning/v2/task_6_13_full_task_14_s_1_50",
    "basic-baseline/test/reasoning/v2/task_14_s_50_100_task_15_16_full",
]
baseline_reasoning_v2 = [
    "basic-baseline/test/reasoning/v2/task_1",
    "basic-baseline/test/reasoning/v2/task_2_3_5_full_task_14_s_1_55",
    "basic-baseline/test/reasoning/v2/task_4",
    "basic-baseline/test/reasoning/v2/task_6_13_full_task_14_s_1_50",
    "basic-baseline/test/reasoning/v2/task_14",
    "basic-baseline/test/reasoning/v2/task_14_s_50_100_task_15_16_full",
    "basic-baseline/test/reasoning/v2/task_17_18",
    "basic-baseline/test/reasoning/v2/task_19",
    "basic-baseline/test/reasoning/v2/task_20",
]
baseline_reasoning_v3_t_18 = [
    "basic-baseline/test/reasoning/v3/task_18_s_1_25",
    "basic-baseline/test/reasoning/v3/task_18_s_25_100",
]
baseline_reasoning_v3 = [
    "basic-baseline/test/reasoning/v3/task_1",
    "basic-baseline/test/reasoning/v3/task_2_3_5_14",
    "basic-baseline/test/reasoning/v3/task_4",
    "basic-baseline/test/reasoning/v3/task_6_16",
    "basic-baseline/test/reasoning/v3/task_17",
    "basic-baseline/test/reasoning/v3/task_18",
    "basic-baseline/test/reasoning/v3/task_19",
    "basic-baseline/test/reasoning/v3/task_20",
]

baseline_reasoning_v4_t_11 = [
    "basic-baseline/test/reasoning/v4/task_6_10_full_task_11_s_1_50",
    "basic-baseline/test/reasoning/v4/task_11_s_50_100_task_12_s_1_35",
]
baseline_reasoning_v4_t_12 = [
    "basic-baseline/test/reasoning/v4/task_11_s_50_100_task_12_s_1_35",
    "basic-baseline/test/reasoning/v4/task_12_s_36_100_task_13_15_16_full_task_18_s_1_24",
]
baseline_reasoning_v4_t_17 = [
    "basic-baseline/test/reasoning/v4/task_2_3_5_14_full_task_17_s_1_20",
    "basic-baseline/test/reasoning/v4/task_17_s_15_100",
]
baseline_reasoning_v4_t_18 = [
    "basic-baseline/test/reasoning/v4/task_12_13_15_16_full_task_18_s_1_24",
    "basic-baseline/test/reasoning/v4/task_18_s_25_100",
]
baseline_reasoning_v4 = [
    "basic-baseline/test/reasoning/v4/task_1_full_task_2_s_1_50",
    "basic-baseline/test/reasoning/v4/task_2_3_5_14_full_task_17_s_1_20",
    "basic-baseline/test/reasoning/v4/task_4",
    "basic-baseline/test/reasoning/v4/task_6_10_full_task_11_s_1_50",
    "basic-baseline/test/reasoning/v4/task_11",
    "basic-baseline/test/reasoning/v4/task_12",
    "basic-baseline/test/reasoning/v4/task_12_s_36_100_task_13_15_16_full_task_18_s_1_24",
    "basic-baseline/test/reasoning/v4/task_17",
    "basic-baseline/test/reasoning/v4/task_18",
    "basic-baseline/test/reasoning/v4/task_19",
    "basic-baseline/test/reasoning/v4/task_20",
]

baseline_reasoning_v5_t_15 = [
    "basic-baseline/test/reasoning/v5/task_15_s_1_75",
    "basic-baseline/test/reasoning/v5/task_15_end_task_16_full",
]
baseline_reasoning_v5_t_17 = [
    "basic-baseline/test/reasoning/v5/task_2_3_5_14_full_task_17_s_1_15",
    "basic-baseline/test/reasoning/v5/task_17_s_15_100",
]
baseline_reasoning_v5 = [
    "basic-baseline/test/reasoning/v5/task_1_task_2_s_1_25",
    "basic-baseline/test/reasoning/v5/task_2_3_5_14_full_task_17_s_1_15",
    "basic-baseline/test/reasoning/v5/task_4",
    "basic-baseline/test/reasoning/v5/task_6",
    "basic-baseline/test/reasoning/v5/task_7_13",
    "basic-baseline/test/reasoning/v5/task_15",
    "basic-baseline/test/reasoning/v5/task_15_end_task_16_full",
    "basic-baseline/test/reasoning/v5/task_17",
    "basic-baseline/test/reasoning/v5/task_18",
    "basic-baseline/test/reasoning/v5/task_19",
    "basic-baseline/test/reasoning/v5/task_20",
]
