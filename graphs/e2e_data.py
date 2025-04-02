labels = {
    "UnifiedPP": "UnifiedPP",
    "Interleaved-L": "Interleaved-L",
    "Interleaved": "Interleaved",
    "ZBV": "ZBV",
    "ZBH": "ZBH",
    "1F1B": "1F1B",
}

data = {
    "56B LLaMA": {
         "pp4 tp4 dp2 mb16":{
            "homo":{
                "UnifiedPP": 535,
                "Interleaved-L": 541,
                "Interleaved": 512,
                "ZBV": 526,
                "ZBH": 506,
                "1F1B": 478,
            },
            "heter":{
                "UnifiedPP": 275,
                "Interleaved-L": 158,
                "Interleaved": 158,
                "ZBV": 255,
                "ZBH": 225,
                "1F1B": 238,
            },
        },
        "pp4 tp8 dp1 mb16":{
            "heter":{
                "UnifiedPP": 184,
                "Interleaved-L": 79,
                "Interleaved": 79,
                "ZBV": 128,
                "ZBH": 114,
                "1F1B": 118,
            },
            "homo":{
                "UnifiedPP": 450,
                "Interleaved-L": 450,
                "Interleaved": 433,
                "ZBV": 420,
                "ZBH": 410,
                "1F1B": 400,
            },
        },
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "UnifiedPP": 490,
                "Interleaved-L": 500,
                "Interleaved": 475,
                "ZBV": 495,
                "ZBH": 475,
                "1F1B": 435,
            },
            "heter":{
                "UnifiedPP": 168,
                "Interleaved-L": 82,
                "Interleaved": 81,
                "ZBV": 134,
                "ZBH": 115,
                "1F1B": 120,
            },
        },
    },
    "70B LLaMA": {
        "pp4 tp4 dp4 mb16":{
            "homo":{
                "UnifiedPP": 415,
                "Interleaved-L": 415,
                "Interleaved": 399,
                "ZBV": 400,
                "ZBH": 390,
                "1F1B": 370,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp2 mb16":{
            "homo":{
                "UnifiedPP": 334,
                "Interleaved-L": 331,
                "Interleaved": 318,
                "ZBV": 307,
                "ZBH": 305,
                "1F1B": 295,
            },
            "heter":{
            },
        },
        "pp8 tp4 dp2 mb32":{
            "homo":{
                "UnifiedPP": 414,
                "Interleaved-L": 410,
                "Interleaved": 390,
                "ZBV": 404,
                "ZBH": 386,
                "1F1B": 356,
            },
            "heter":{
            },
        },
        # "pp8 tp8 dp1 mb32":{
        #     "homo":{
        #         "UnifiedPP": 348,
        #         "Interleaved-L": 347,
        #         "Interleaved": 328,
        #         "ZBV": 330,
        #         "ZBH": 316,
        #         "1F1B": 300,
        #     },
        #     "heter":{
        #     },
        # },
    },
}

data_h800 = {
    "70B LLaMA": {
         "pp4 tp4 dp2 mb16":{
            "homo":{
                "UnifiedPP": 761,
                "Interleaved-L": 660,
                "Interleaved": 200,
                "ZBV": 0,
                "ZBH": 0,
                "1F1B": 670,
            },
            "heter":{
                "UnifiedPP": 0,
                "Interleaved-L": 0,
                "Interleaved": 0,
                "ZBV": 0,
                "ZBH": 0,
                "1F1B": 0,
            },
        },
        "pp4 tp8 dp1 mb16":{
            "homo":{
                "UnifiedPP": 599,
                "Interleaved-L": 608,
                "Interleaved": 570,
                "ZBV": 534,
                "ZBH": 553,
                "1F1B": 540,
            },
            "heter":{
                "UnifiedPP": 0,
                "Interleaved-L": 0,
                "Interleaved": 0,
                "ZBV": 0,
                "ZBH": 0,
                "1F1B": 0,
            },
        },
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "UnifiedPP": 790,
                "Interleaved-L": 801,
                "Interleaved": 400,
                "ZBV": 775,
                "ZBH": 750,
                "1F1B": 720,
            },
            "heter":{
                "UnifiedPP": 0,
                "Interleaved-L": 0,
                "Interleaved": 0,
                "ZBV": 0,
                "ZBH": 0,
                "1F1B": 0,
            },
        },
    },
    "42B LLaMA": {
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "UnifiedPP": 1260,
                "Interleaved-L": 1245,
                "Interleaved": 1207,
                "ZBV": 1040,
                "ZBH": 1180,
                "1F1B": 1128,
            },
            "heter":{
            },
        },
        "pp4 tp4 dp2 mb16":{
            "homo":{
                "UnifiedPP": 1335,
                "Interleaved-L": 1363,
                "Interleaved": 1325,
                "ZBV": 1275,
                "ZBH": 1270,
                "1F1B": 1226,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp1 mb16":{
            "homo":{
                "UnifiedPP": 955,
                "Interleaved-L": 970,
                "Interleaved": 0,
                "ZBV": 859,
                "ZBH": 774,
                "1F1B": 743,
            },
            "heter":{
            },
        },
        "pp4 tp4 dp1 mb16":{
            "homo":{
                "UnifiedPP": 1291,
                "Interleaved-L": 1311,
                "Interleaved": 1277,
                "ZBV": 1271,
                "ZBH": 1240,
                "1F1B": 1200,
            },
            "heter":{
            },
        },
        "pp8 tp2 dp1 mb32":{
            "homo":{
                "UnifiedPP": 1395,
                "Interleaved-L": 0,
                "Interleaved": 0,
                "ZBV": 0,
                "ZBH": 0,
                "1F1B": 1334,
            },
            "heter":{
            },
        },
    },
    "14B LLaMA": {
        "pp4 tp2 dp1 mb16":{
            "homo":{
                "UnifiedPP": 3776,
                "Interleaved-L": 3693,
                "Interleaved": 3693,
                "ZBV": 3615,
                "ZBH": 3684,
                "1F1B": 3475,
            },
            "heter":{
            },
        },
        "pp8 tp1 dp1 mb32":{
            "homo":{
                "UnifiedPP": 3455,
                "Interleaved-L": 3262,
                "Interleaved": 3262,
                "ZBV": 0,
                "ZBH": 3288,
                "1F1B": 3130,
            },
            "heter":{
            },
        },
    },
}

colors = {
    "UnifiedPP": "#F8CECC",
    "Interleaved-L": "#DAE8FC",
    "Interleaved": "#99CCFF",
    "ZBV": "#D5E8D4",
    "ZBH": "#FFE6CC",
    "1F1B": "#FFF2CC",
}

ablation_study_data = {
    "70B LLaMA": {
         "pp4 tp4 dp2 mb16":{
            "homo":{
                "+ Overlap-aware": 761,
                "+ Memory monitor": 741,
                "+ Adaptive workload schedule": 742,
                "+ Adaptive model placement": 718,
                "1F1B": 670,
            },
            "heter":{
            },
        },
    },
    "42B LLaMA": {
        "pp4 tp8 dp1 mb16":{
            "homo":{
                "+ Overlap-aware": 950,
                "+ Memory monitor": 922,
                "+ Adaptive workload schedule": 921,
                "+ Adaptive model placement": 841,
                "1F1B": 745,
            },
            "heter":{
            },
        },
    },
    "14B LLaMA": {
        "pp8 tp1 dp1 mb32":{
            "homo":{
                "+ Overlap-aware": 3402,
                "+ Memory monitor": 3356,
                "+ Adaptive workload schedule": 0,
                "+ Adaptive model placement": 3301,
                "1F1B": 3124,
            },
            "heter":{
            },
        },
    },
}

sim = {
    "56B LLaMA": {
         "pp4 tp4 dp2 mb16":{
            "homo":{
                "UnifiedPP": 1624,
                "Interleaved-L": 1634,
                "Interleaved": 1760,
                "ZBV": 1663,
                "ZBH": 1712,
                "1F1B": 1904,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp1 mb16":{
            "heter":{
            },
            "homo":{
                "UnifiedPP": 1624,
                "Interleaved-L": 1634,
                "Interleaved": 1760,
                "ZBV": 1663,
                "ZBH": 1712,
                "1F1B": 1904,
            },
        },
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "UnifiedPP": 1744,
                "Interleaved-L": 1738,
                "Interleaved": 1864,
                "ZBV": 1751,
                "ZBH": 1808,
                "1F1B": 2032,
            },
        },
    },
    "70B LLaMA": {
        "pp4 tp4 dp4 mb16":{
            "homo":{
                "UnifiedPP": 2039,
                "Interleaved-L": 2018,
                "Interleaved": 2180,
                "ZBV": 2059,
                "ZBH": 2120,
                "1F1B": 2360,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp2 mb16":{
            "homo":{
                "UnifiedPP": 2039,
                "Interleaved-L": 2018,
                "Interleaved": 2180,
                "ZBV": 2059,
                "ZBH": 2120,
                "1F1B": 2360,
            },
            "heter":{
            },
        },
        "pp8 tp4 dp2 mb32":{
            "homo":{
                "UnifiedPP": 2122,
                "Interleaved-L": 2122,
                "Interleaved": 2290,
                "ZBV": 2149,
                "ZBH": 2220,
                "1F1B": 2500,
            },
            "heter":{
            },
        },
    },
    "14B LLaMA": {
        "pp8 tp1 dp1 mb32":{
            "homo":{
                "UnifiedPP": 2332,
                "Interleaved-L": 2386,
                "Interleaved": 2386,
                "ZBV": 2283,
                "ZBH": 2292,
                "1F1B": 2462,
            },
            "heter":{
            },
        },
        "pp4 tp2 dp1 mb16":{
            "homo":{
                "UnifiedPP": 1698,
                "Interleaved-L": 1796,
                "Interleaved": 1796,
                "ZBV": 1726,
                "ZBH": 1746,
                "1F1B": 1898,
            },
            "heter":{
            },
        },
    },
}

sim_h800 = {
    "70B LLaMA": {
        "pp4 tp4 dp2 mb16":{
            "homo":{
                "UnifiedPP": 6383,
                "Interleaved-L": 6350,
                "Interleaved": 6836,
                "ZBV": 6454,
                "ZBH": 6738,
                "1F1B": 7370,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp1 mb16":{
            "homo":{
                "UnifiedPP": 5802,
                "Interleaved-L": 5814,
                "Interleaved": 6300,
                "ZBV": 6454,
                "ZBH": 6120,
                "1F1B": 6840,
            },
            "heter":{
            },
        },
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "UnifiedPP": 2122,
                "Interleaved-L": 8056,
                "Interleaved": 8332,
                "ZBV": 7031,
                "ZBH": 7314,
                "1F1B": 8078,
            },
            "heter":{
            },
        },
    },
    "42B LLaMA": {
        "pp4 tp4 dp1 mb16":{
            "homo":{
                "UnifiedPP": 4002,
                "Interleaved-L": 4046,
                "Interleaved": 4316,
                "ZBV": 4090,
                "ZBH": 4242,
                "1F1B": 4634,
            },
            "heter":{
            },
        },
        "pp4 tp4 dp2 mb16":{
            "homo":{
                "UnifiedPP": 4002,
                "Interleaved-L": 4046,
                "Interleaved": 4316,
                "ZBV": 4090,
                "ZBH": 4242,
                "1F1B": 4634,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp1 mb16":{
            "homo":{
                "UnifiedPP": 4002,
                "Interleaved-L": 4046,
                "Interleaved": 4316,
                "ZBV": 4090,
                "ZBH": 4242,
                "1F1B": 4634,
            },
            "heter":{
            },
        },
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "UnifiedPP": 4554,
                "Interleaved-L": 5176,
                "Interleaved": 4620,
                "ZBV": 4657,
                "ZBH": 4786,
                "1F1B": 5270,
            },
            "heter":{
            },
        },
    },
    "14B LLaMA": {
        "pp8 tp1 dp1 mb32":{
            "homo":{
                "UnifiedPP": 2332,
                "Interleaved-L": 2386,
                "Interleaved": 2386,
                "ZBV": 2283,
                "ZBH": 2292,
                "1F1B": 2462,
            },
            "heter":{
            },
        },
        "pp4 tp2 dp1 mb16":{
            "homo":{
                "UnifiedPP": 1698,
                "Interleaved-L": 1796,
                "Interleaved": 1796,
                "ZBV": 1726,
                "ZBH": 1746,
                "1F1B": 1898,
            },
            "heter":{
            },
        },
    },
}

mem_pro_h800 = {
    "14B LLaMA": {
        "pp8 tp1 dp1 mb32":{
            "homo":{
                "UnifiedPP": [66.4321, 49.3999, 49.1665, 48.0542, 46.8794, 45.813, 44.7632, 68.9313 - 5], #TODO runtime wrong order
                "Interleaved-L": [66.103, 47.5874, 45.2378, 42.8882, 40.5386, 38.189, 35.8394, 52.5878],
                "Interleaved": [66.103, 47.5874, 45.2378, 42.8882, 40.5386, 38.189, 35.8394, 52.5878],
                "ZBV": ["OOM","OOM","OOM","OOM","OOM","OOM","OOM","OOM"],
                "ZBH": [56.7446, 41.4009, 41.1216, 40.8423, 40.563, 40.2837, 40.0044, 66.2174],
                "1F1B": [56.7446, 38.8638, 36.5767, 34.2896, 32.0025, 29.7153, 27.7252, 47.2957],
            },
            "heter":{
            },
        },
    },
}
mem_sim_h800 = {
    "14B LLaMA": {
        "pp8 tp1 dp1 mb32":{
            "homo":{
                "UnifiedPP": [67.042, 49.749, 49.459, 48.35, 47.24, 46.421, 45.311, 62.975],
                "Interleaved-L": [65.296, 47.421, 45.202, 42.984, 40.765, 38.547, 36.328, 53.18],
                "Interleaved": [65.296, 47.421, 45.202, 42.984, 40.765, 38.547, 36.328, 53.18],
                "ZBV": [79.841, 44.093, 44.093, 44.093, 45.202, 45.202, 46.311, 46.311],
                "ZBH": [59.346, 43.732, 43.153, 42.573, 41.994, 41.415, 40.836, 67.698],
                "1F1B": [57.531, 39.656, 37.437, 35.219, 33.0, 30.781, 28.563, 45.852],
            },
            "heter":{
            },
        },
    },
}


# from preprocess import generate_
# MEMORY_PROFILE_ALL = False
# JOB_NAME = "7b_llama2_train"
# model_type = "LLAMA2"
# DO_ALERT = False
# layerwise = False
# VOCAB_SIZE = 128256
# SEQ_LEN = 4096
# HIDDEN_SIZE = 8192
# NUM_ATTENTION_HEAD = 64
# NUM_KV_ATTENTION_HEAD = 32
# MLP_RATIO = 2.6875
# NUM_LAYER = 48
# PP_SIZE = 8
# TP_SIZE = 4
# ZERO_SZIE = 1
# MICRO_NUM = PP_SIZE * 4
# STEP = 10

# SCHEDULE = 1
# if SCHEDULE == 0:
#     num_microbatches, pp_size, stage_placement, scheduler_type ,\
#     split_backward, unified_scheduler, comm_graph, first_stage ,\
#     last_stage, Devices_containing_last_stage, chunk_num = generate_()
# PP_MODE = None
# CHUNK_NUM = None
# if SCHEDULE == 0: # upp
#     PP_MODE = "unified"
#     CHUNK_NUM = chunk_num
#     PP_SIZE = pp_size
#     MICRO_NUM = num_microbatches
# elif SCHEDULE == 1: # i1f1b-l
#     PP_MODE = "1f1b"
#     CHUNK_NUM = NUM_LAYER // PP_SIZE
# elif SCHEDULE == 2: # i1f1b
#     PP_MODE = "1f1b"
#     CHUNK_NUM = 2
# elif SCHEDULE == 3: # zbv
#     PP_MODE = "zbv"
#     CHUNK_NUM = 1
# elif SCHEDULE == 4: # zbh1
#     PP_MODE = "zbh1"
#     CHUNK_NUM = 1
# elif SCHEDULE == 5: # 1f1b
#     PP_MODE = "1f1b"
#     CHUNK_NUM = 1
# else:
#     raise ValueError("Wrong Schedule.")

# print(f"{PP_MODE},CHUNK{CHUNK_NUM},LAYER{NUM_LAYER},PP{PP_SIZE},TP{TP_SIZE},DP{ZERO_SZIE},NMB{MICRO_NUM}")
# MODEL_ONLY_FOLDER = "local:llm_ckpts/xxxx"
# # Ckpt folder format:
# # fs: 'local:/mnt/nfs/XXX'
# SAVE_CKPT_FOLDER = "local:llm_ckpts"
# LOAD_CKPT_FOLDER = "local:llm_ckpts/49"

# # boto3 Ckpt folder format:
# # import os
# # BOTO3_IP = os.environ["BOTO3_IP"] # boto3 bucket endpoint
# # SAVE_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm"
# # LOAD_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm/snapshot/1/"
# CHECKPOINT_EVERY = 50
# ckpt = dict(
#     enable_save_ckpt=False,  # enable ckpt save.
#     save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
#     # 'auto_resume' is designed to automatically load the latest checkpoint from 'save_ckpt_folder' when encountering
#     # training interruptions/hangs caused by hardware failures, using a scheduling system (such as k8s/slurm)
#     # with an automatic restart mechanism upon training reboot.
#     # Please be aware that if `auto_resume` is not set (its default value is True), it will not load the checkpoint
#     # path specified in `load_ckpt_info` by default.
#     # If you want to initialize your model weights from another model, you must set `auto_resume` to False.
#     # If you want to train from scratch, please set `auto_resume` to False and 'load_ckpt_info' to None.
#     auto_resume=False,
#     checkpoint_every=CHECKPOINT_EVERY,
#     async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
#     async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
#     oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
# )

# TRAIN_FOLDER = None
# VALID_FOLDER = None  # "/path/to/dataset"
# data = dict(
#     seq_len=SEQ_LEN,
#     # micro_num means the number of micro_batch contained in one gradient update
#     micro_num=MICRO_NUM,
#     # packed_length = micro_bsz * SEQ_LEN
#     micro_bsz=1,
#     # defaults to the value of micro_num
#     valid_micro_num=4,
#     # defaults to 0, means disable evaluate
#     valid_every=0,
#     pack_sample_into_one=False,
#     total_steps=STEP,
#     skip_batches="",
#     # rampup_batch_size (str): A string with three space-separated integers representing the
#     #       starting batch size, the increment, and the number of steps between
#     #       each increment. For example, "192 24 8" means that the batch size (micro_num)
#     #       starts at 192 and increases by 24 every 8 steps. Defaults to None.
#     #       (IMPORTANT): The interval step size is 'micro_bsz'.
#     rampup_batch_size="",
#     # Datasets with less than 50 rows will be discarded
#     min_length=50,
#     train_folder=TRAIN_FOLDER,
#     valid_folder=VALID_FOLDER,
#     empty_cache_and_diag_interval=200,
#     diag_outlier_ratio=1.1,
# )

# grad_scaler = dict(
#     fp16=dict(
#         # the initial loss scale, defaults to 2**16
#         initial_scale=2**16,
#         # the minimum loss scale, defaults to None
#         min_scale=1,
#         # the number of steps to increase loss scale when no overflow occurs
#         growth_interval=1000,
#     ),
#     # the multiplication factor for increasing loss scale, defaults to 2
#     growth_factor=2,
#     # the multiplication factor for decreasing loss scale, defaults to 0.5
#     backoff_factor=0.5,
#     # the maximum loss scale, defaults to None
#     max_scale=2**24,
#     # the number of overflows before decreasing loss scale, defaults to 2
#     hysteresis=2,
# )

# hybrid_zero_optimizer = dict(
#     # Enable low_level_optimzer overlap_communication
#     overlap_sync_grad=False,
#     overlap_sync_param=False,
#     # bucket size for nccl communication params
#     reduce_bucket_size=512 * 1024 * 1024,
#     # grad clipping
#     clip_grad_norm=1.0,
# )

# loss = dict(
#     label_smoothing=0,
# )

# adam = dict(
#     lr=1e-4,
#     adam_beta1=0.9,
#     adam_beta2=0.95,
#     adam_beta2_c=0,
#     adam_eps=1e-8,
#     weight_decay=0.01,
# )

# lr_scheduler = dict(
#     total_steps=data["total_steps"],
#     init_steps=0,  # optimizer_warmup_step
#     warmup_ratio=0.01,
#     eta_min=1e-5,
#     last_epoch=-1,
# )

# beta2_scheduler = dict(
#     init_beta2=adam["adam_beta2"],
#     c=adam["adam_beta2_c"],
#     cur_iter=-1,
# )

# use_fp32_norm = False
# model = dict(
#     checkpoint=False,
#     num_chunks=CHUNK_NUM,
#     num_attention_heads=NUM_ATTENTION_HEAD,
#     embed_split_hidden=True,
#     vocab_size=VOCAB_SIZE,
#     embed_grad_scale=1,
#     parallel_output=True,
#     hidden_size=HIDDEN_SIZE,
#     num_layers=NUM_LAYER,
#     no_bias=True,
#     mlp_ratio=MLP_RATIO,
#     apply_post_layer_norm=False,
#     dtype="torch.bfloat16",
#     norm_type="rmsnorm",
#     layer_norm_epsilon=1e-5,
#     num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
#     use_flash_attn=True,
#     # Whether the odd and even columns of the query and key in the model are normally interleaved.
#     # If it's True, the model's odd and even columns are normally ordered; if it's False,
#     # it means that the model has prematurely concatenated all odd columns and even columns in front
#     # and back, in order to improve the RoPE's computational efficiency.
#     # Example:
#     # qk_interleaved = True: q[-1] = [q1,q2,q3,q4,q5,q6,...], k[-1] = [k1,k2,k3,k4,k5,k6,...]
#     # qk_interleaved = False: q[-1] = [q1,q3,q5,...,q2,q4,q6,...], k[-1] = [k1,k3,k5,...,k2,k4,k6,...]
#     qk_interleaved=False,
#     mlp_layer_fusion=True,
#     enable_qkv_fusion=True,
# )

# """
# zero1 parallel (dict):
#     1. size: int
#         * if size <= 0, the size of the zero process group is equal to the size of the dp process group,
#             so parameters will be divided within the range of dp.
#         * if size == 1, zero is not used, and all dp groups retain the full amount of model parameters.
#         * if size > 1 and size <= dp world size, the world size of zero is a subset of dp world size.
#         For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
#     2. fsdp: bool, enable/disable torch's fully sharded data parallel, defaults to False.
# tensor parallel (dict):
#     1. size: int, the size of tensor parallel.
#     2. mode: str, the tensor parallel mode, should be in ['mtp', 'msp', 'fsp', 'isp'],
#         defaults to 'mtp', means the pure megatron tensor parallel without sequence parallel.
#         msp: megatron tensor parallel with sequence parallel, sequence parallel size = tensor parallel size.
#         fsp: tensor parallel by flash-attn with sequence parallel, sequence parallel size = tensor parallel size.
#         isp: customed intern sequence parallel without tensor parallel, can be used with weight parallel.
# pipeline parallel (dict):
#     1. size: int, the size of pipeline parallel.
#     2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler,
#         defaults to False.
# weight parallel (dict):
#     1. size: int, the size of weight parallel.
#     2. overlap: bool, enable/disable all_gather/reduce_scatter communication overlap, defaults to False.
# """
# parallel = dict(
#     zero1=dict(size=ZERO_SZIE),
#     tensor=dict(size=TP_SIZE, mode="fsp"),
#     pipeline=dict(size=PP_SIZE, interleaved_overlap=True, mode=PP_MODE),
#     weight=dict(size=1, overlap=True),
# )

# cudnn_deterministic = False
# cudnn_benchmark = False

# monitor = dict(
#     # feishu alert configs
#     alert=dict(
#         enable_feishu_alert=DO_ALERT,
#         feishu_alert_address=None,  # feishu webhook to send alert message
#         light_monitor_address=None,  # light_monitor address to send heartbeat
#         alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
#     ),
#     tensorboard=dict(
#         queue_max_length=10,
#     ),
# )

# # metric_dtype can be "fp32" or other string
# # only when set to "fp32" will use fp32 to calc in metrics
# # metric_dtype = "fp32"