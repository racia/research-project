sd_reasoning_v1_t_1 = [
    "SD/test/reasoning/v1/unjoined/task_1/task_1_s_1_84",
    "SD/test/reasoning/v1/unjoined/task_1/task_1_s_84_100",
]
sd_reasoning_v1_t_2 = [
    "SD/test/reasoning/v1/unjoined/task_2/task_2_s_1_76",
    "SD/test/reasoning/v1/unjoined/task_2/task_2_s_76_100",
]
sd_reasoning_v1_t_3 = [
    "SD/test/reasoning/v1/unjoined/task_3/task_3_s_1_51",
    "SD/test/reasoning/v1/unjoined/task_3/task_3_s_51_59",
    "SD/test/reasoning/v1/unjoined/task_3/task_3_s_59_86",
    "SD/test/reasoning/v1/unjoined/task_3/task_3_s_86_100",
]
sd_reasoning_v1_t_5 = [
    # TODO: samples 1-60 are absent
    "SD/test/reasoning/v1/unjoined/task_5/task_5_s_61_89",
    "SD/test/reasoning/v1/unjoined/task_5/task_5_s_89_100",
]
sd_reasoning_v1_t_6 = [
    "SD/test/reasoning/v1/unjoined/task_6/task_6_s_1_21",
    "SD/test/reasoning/v1/unjoined/task_6/task_6_s_21_100",
]
sd_reasoning_v1_t_7 = [
    "SD/test/reasoning/v1/unjoined/task_7/task_7_s_1_92",
    "SD/test/reasoning/v1/unjoined/task_7/task_7_s_92_100",
]
sd_reasoning_v1_t_8 = [
    "SD/test/reasoning/v1/unjoined/task_8/task_8_s_1_59",
    "SD/test/reasoning/v1/unjoined/task_8/task_8_s_59",
    "SD/test/reasoning/v1/unjoined/task_8/task_8_s_60_100",
]
sd_reasoning_v1_t_9 = [
    "SD/test/reasoning/v1/unjoined/task_9/task_9_s_1_78",
    "SD/test/reasoning/v1/unjoined/task_9/task_9_s_78_83",
    "SD/test/reasoning/v1/unjoined/task_9/task_9_s_83_100",
]
sd_reasoning_v1_t_10 = [
    "SD/test/reasoning/v1/unjoined/task_10/task_10_s_1_88",
    "SD/test/reasoning/v1/unjoined/task_10/task_10_s_88_100",
]
sd_reasoning_v1_t_11 = [
    "SD/test/reasoning/v1/unjoined/task_11/task_11_s_1_89",
    "SD/test/reasoning/v1/unjoined/task_11/task_11_s_89_100",
]
sd_reasoning_v1_t_12 = [
    "SD/test/reasoning/v1/unjoined/task_12/task_12_s_1_93",
    "SD/test/reasoning/v1/unjoined/task_12/task_12_s_93_100",
]
sd_reasoning_v1_t_13 = [
    "SD/test/reasoning/v1/unjoined/task_13/task_13_s_1_89",
    "SD/test/reasoning/v1/unjoined/task_13/task_13_s_89_93",
    "SD/test/reasoning/v1/unjoined/task_13/task_13_s_93_100",
]
sd_reasoning_v1_t_14 = [
    "SD/test/reasoning/v1/unjoined/task_14/task_14_s_1_60",
    "SD/test/reasoning/v1/unjoined/task_14/task_14_s_59_100",
]
sd_reasoning_v1_t_17 = [
    "SD/test/reasoning/v1/unjoined/task_17/task_17_s_1_43",
    "SD/test/reasoning/v1/unjoined/task_17/task_17_s_43_84",
    "SD/test/reasoning/v1/unjoined/task_17/task_17_s_84_97",
    "SD/test/reasoning/v1/unjoined/task_17/task_17_s_97_100",
]
sd_reasoning_v1_t_18 = [
    "SD/test/reasoning/v1/unjoined/task_18/task_18_s_1_84",
    "SD/test/reasoning/v1/unjoined/task_18/task_18_s_84_100",
]
sd_reasoning_v1_t_20 = [
    "SD/test/reasoning/v1/unjoined/task_20/task_20_s_1_12",
    "SD/test/reasoning/v1/unjoined/task_20/task_20_s_10_21",
    "SD/test/reasoning/v1/unjoined/task_20/task_20_s_20_27",
    "SD/test/reasoning/v1/unjoined/task_20/task_20_s_27_62",
    "SD/test/reasoning/v1/unjoined/task_20/task_20_s_62_91",
    "SD/test/reasoning/v1/unjoined/task_20/task_20_s_91",
    "SD/test/reasoning/v1/unjoined/task_20/task_20_s_92_93",
]
sd_reasoning_v1 = [
    (
        f"SD/test/reasoning/v1/task_{i}"
        if i != 5
        else f"SD/test/reasoning/v1/task_{i}_full"
    )
    for i in range(1, 21)
]
