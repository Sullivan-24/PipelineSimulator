labels = {
    "UnifiedPP": "UnifiedPP",
    "Interleaved-L": "Interleaved-L",
    "Interleaved": "Interleaved",
    "ZBV": "ZBV",
    "ZBH": "ZBH",
    "1F1B": "1F1B",
}

colors = {
    "UnifiedPP": "#F8CECC",
    "Interleaved-L": "#DAE8FC",
    "Interleaved": "#99CCFF",
    "ZBV": "#D5E8D4",
    "ZBH": "#FFE6CC",
    "1F1B": "#FFF2CC",
}

layer_overhead = {
    "Megatron":{
        "F" : {
            "Transformer layer":(21.684+21.225+21.458+22.449) / 4,
            "+ Non-transformer layer": (58.342+59.566+58.005+60.162) / 4,
        },
        "B" : {
            "Transformer layer":(45.983+45.057+46.229+45.409) / 4,
            "+ Non-transformer layer": (78.800+76.587+77.652+74.893) / 4,
        },
    },
    "Backward-splitting":{
        "F" : {
            "Transformer layer":(20.998+20.882+20.518+21.359) / 4,
            "+ Non-transformer layer": (56.983+57.393+57.195+57.078) / 4,
        },
        "B" : {
            "Transformer layer":(26.997+28.427+27.338+27.372) / 4,
            "+ Non-transformer layer": (45.929+45.768+45.422+46.298) / 4,
        },
        "W" : {
            "Transformer layer":(18.869+18.340+18.879+18.341) / 4,
            "+ Non-transformer layer": (33.906+32.943+33.697+33.066) / 4,
        },
    },
}

preprocessing = {
    "70B LLaMA": {
        # "pp2":{
        #     "single":13.459882974624634/5,
        #     "multi":13.459882974624634/5,
        # },
        "pp4":{
            "single":39.11562919616699/5,
            "multi":39.11562919616699/5,
        },
        "pp8":{
            "single":114.67990231513977/5,
            "multi":114.67990231513977/5,
        },
        "pp10":{
            "single": 56.93681597709656,
            "multi": 56.93681597709656,
        },
        "pp16":{
            "single": 779.8600070476532/5,
            "multi": 779.8600070476532/5,
        },
        "pp20":{
            "single": 224.93572807312012,
            "multi": 224.93572807312012,
        },
        "pp40":{
            "single": 1045.2018249034882,
            "multi": 1045.2018249034882,
        },
    },
    "42B LLaMA": {
        # "pp2":{
        #     "single":5.076202869415283/5,
        #     "multi":5.076202869415283/5,
        # },
        "pp4":{
            "single":17.601519107818604/5,
            "multi":17.601519107818604/5,
        },
        "pp8":{
            "single":67.80011010169983/5,
            "multi":67.80011010169983/5,
        },
        "pp12":{
            "single":158.8371820449829/5,
            "multi":158.8371820449829/5,
        },
        "pp16":{
            "single":301.7992088794708/5,
            "multi":301.7992088794708/5,
        },
        "pp24":{
            "single":133.84632897377014,
            "multi":133.84632897377014,
        }
    },
    "14B LLaMA": {
        # "pp2":{
        #     "single":0.756459712982177/5,
        #     "multi":0.756459712982177/5,
        # },
        "pp4":{
            "single":2.7287838459014893/5,
            "multi":2.7287838459014893/5,
        },
        "pp8":{
            "single":12.119674921035767/5,
            "multi":12.119674921035767/5,
        },
        "pp16":{
            "single":45.608559131622314/5,
            "multi":45.608559131622314/5,
        },
    }
}

comp_data = {
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
                "Interleaved": 565,
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
                "Interleaved": 920,
                "ZBV": 855,
                "ZBH": 890,
                "1F1B": 843,
            },
            "heter":{
            },
        },
        "pp4 tp4 dp1 mb16":{
            "homo":{
                "UnifiedPP": 1309,
                "Interleaved-L": 1311,
                "Interleaved": 1277,
                "ZBV": 1271,
                "ZBH": 1240,
                "1F1B": 1195,
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
