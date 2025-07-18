labels = {
    "OctoPipe": "OctoPipe",
    "Interleaved": "Interleaved",
    "ZBV": "ZBV",
    "ZBH": "ZBH",
    "ZB": "ZB",
    "Dapple": "Dapple",
    "Naive": "Naive",
    "S-1F1B": "S-1F1B",
    "I-1F1B": "I-1F1B",
}

hatches = {
    "OctoPipe": "//",
    "Interleaved": "+",
    "ZBV": "x",
    "ZBH": "/",
    "ZB": "x",
    "Dapple": ".",
    "S-1F1B": ".",
    "I-1F1B": "\\\\",
    "Naive": ".",
    "Alpa": "+",
    "Mist": "+",
    "Metis": "//",
}

heter_labels = {
    "OctoPipe": "OctoPipe",
    "Alpa": "Alpa",
    "Mist": "Mist",
    "Metis": "Metis",
    "Dapple": "Dapple",
    "Naive": "Naive",
    "S-1F1B": "S-1F1B",
    "I-1F1B": "I-1F1B",
}

colors = {
    "OctoPipe": "#E54C5E",
    "Interleaved": "#E1D5E7",
    "I-1F1B": "#E1D5E7",
    "ZBV": "#FFE6CC",
    "ZBH": "#D5E8D4",
    "ZB": "#4A90E2",
    "Dapple": "#ffde82",
    "S-1F1B": "#ffde82",
    "Naive": "#ffde82",
    # "Metis": "#E1D5E7",
    "Metis": "#99CCFF",
    # "Alpa": "#B0E3E6",
    "Alpa": "#48BEA3",
}

model_heter_4_stage = {
    "Gemma" : {
        "2K":{
            "fwd": [0.030979193173922025, 0.030027713094438826, 0.03018945058186849, 0.05832540988922119],
            "bwd": [0.06091987169705904, 0.05745237214224679, 0.05634080568949382, 0.07649827003479004],
            "mem": [36.2285, 26.8379, 22.8945, 29.0117],
        },
        "4K":{
            "fwd": [0.048637921993549056, 0.04729163646697998, 0.04735566775004069, 0.09939318895339966],
            "bwd": [0.0927570049579327, 0.08960611479622978, 0.08855722745259603, 0.12593291699886322],
            "mem": [52.3594, 38.9863, 31.1211, 37.8848],
        }
    },
    # "DeepSeek-V2-RS" : {
    #     "2K":{
    #         "fwd": [0.02848052978515625, 0.06712150573730469, 0.06682658195495605, 0.10119485855102539],
    #         "bwd": [0.053653717041015625, 0.08961176872253418, 0.09766578674316406, 0.12091898918151855],
    #         "mem": [21.8613, 17.3262, 17.3262, 20.4902]
    #     },
    #     "4K":{
    #         "fwd": [0.04205322265625, 0.1081399917602539, 0.10775518417358398, 0.1421959400177002],
    #         "bwd": [0.08239436149597168, 0.15581035614013672, 0.14588403701782227, 0.17009544372558594],
    #         "mem": [32.3164, 29.4632, 27.2215, 28.6402]
    #     }
    # },
    
    "LLaMA3-VF" : {
        "2K":{
            "fwd": [0.022709369659423828, 0.025893211364746094, 0.029953343527657644, 0.05251419544219971],
            "bwd": [0.04073381423950195, 0.045702497164408364, 0.05459431239536831, 0.08169332146644592],
            "mem": [16.8223, 15.7773, 19.0879, 26.7344],
        },
        "4K":{
            "fwd": [0.029047727584838867, 0.036360979080200195, 0.04280996322631836, 0.0858004093170166],
            "bwd": [0.056850433349609375, 0.06891822814941406, 0.0832068920135498, 0.12211060523986816],
            "mem": [22.1387, 21.6699, 24.9531, 33.625],
        }
    },
    "DeepSeek-RS" : {
        "2K": {
            "fwd": [0.018987019856770833, 0.04474767049153646, 0.044551054636637366, 0.06746323903401693],
            "bwd": [0.03576914469401042, 0.05974117914835612, 0.0651105244954427, 0.08061265945435696],
            "mem": [21.8613, 17.3262, 17.3262, 20.4902]
        },
        "4K": {
            "fwd": [0.028035481770833332, 0.07209332784016927, 0.07183678944905598, 0.09479729334513347],
            "bwd": [0.05492957433064779, 0.10387357076009115, 0.0972560246785482, 0.11339696248372395],
            "mem": [32.3164, 29.4632, 27.2215, 28.6402]
        }
    },
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
    "70B": {
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
    "42B": {
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
    "14B": {
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
                "OctoPipe": 535,
                "Interleaved": 512,
                "ZBV": 526,
                "ZBH": 506,
                "Dapple": 478,
            },
            "heter":{
                "OctoPipe": 275,
                "Interleaved": 158,
                "ZBV": 255,
                "ZBH": 225,
                "Dapple": 238,
            },
        },
        "pp4 tp8 dp1 mb16":{
            "heter":{
                "OctoPipe": 184,
                "Interleaved": 79,
                "ZBV": 128,
                "ZBH": 114,
                "Dapple": 118,
            },
            "homo":{
                "OctoPipe": 450,
                "Interleaved": 433,
                "ZBV": 420,
                "ZBH": 410,
                "Dapple": 400,
            },
        },
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "OctoPipe": 490,
                "Interleaved": 475,
                "ZBV": 495,
                "ZBH": 475,
                "Dapple": 435,
            },
            "heter":{
                "OctoPipe": 168,
                "Interleaved": 81,
                "ZBV": 134,
                "ZBH": 115,
                "Dapple": 120,
            },
        },
    },
    "70B LLaMA": {
        "pp4 tp4 dp4 mb16":{
            "homo":{
                "OctoPipe": 415,
                "Interleaved": 399,
                "ZBV": 400,
                "ZBH": 390,
                "Dapple": 370,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp2 mb16":{
            "homo":{
                "OctoPipe": 334,
                "Interleaved": 318,
                "ZBV": 307,
                "ZBH": 305,
                "Dapple": 295,
            },
            "heter":{
            },
        },
        "pp8 tp4 dp2 mb32":{
            "homo":{
                "OctoPipe": 414,
                "Interleaved": 390,
                "ZBV": 404,
                "ZBH": 386,
                "Dapple": 356,
            },
            "heter":{
            },
        },
        # "pp8 tp8 dp1 mb32":{
        #     "homo":{
        #         "OctoPipe": 348,
        #         "Interleaved": 328,
        #         "ZBV": 330,
        #         "ZBH": 316,
        #         "Dapple": 300,
        #     },
        #     "heter":{
        #     },
        # },
    },
}

data_h800_heter = {
    "70B LLaMA": {
        "pp8 tp4 dp1 mb32":{
            "OctoPipe": 390,
            "Alpa": 0,
            "Metis": 0,
            "Dapple": 324,
        },
        "pp4 tp8 dp1 mb16":{
            "OctoPipe": 390,
            "Alpa": 338,
            "Metis": 340,
            "Dapple": 259,
        },
    },
    "42B LLaMA": {
        "pp8 tp4 dp1 mb32":{
            "OctoPipe": 518,
            "Alpa": 472,
            "Metis": 446,
            "Dapple": 363,
        },
        "pp4 tp8 dp1 mb16":{
            "OctoPipe": 655,
            "Alpa": 625,
            "Metis": 505,
            "Dapple": 445,
        },
    },
    "14B LLaMA": {
        "pp8 tp1 dp1 mb32":{
            "OctoPipe": 2965,
            "Alpa": 2156,
            "Metis": 0,
            "Dapple": 1655,
        },
        "pp4 tp2 dp1 mb16":{
            "OctoPipe": 2965,
            "Alpa": 2120,
            "Metis": 2037,
            "Dapple": 1752,
        },
    },
}

data_h800 = {
    "70B LLaMA": {
         "pp4 tp4 dp2 mb16":{
            "homo":{
                "OctoPipe": 761,
                "Interleaved": 200,
                "ZBV": 0,
                "ZBH": 0,
                "Dapple": 670,
            },
            "heter":{
                "OctoPipe": 8563,
                "Interleaved": 0, #13298,
                "ZBV": 0, #13072,
                "ZBH": 0, #13072,
                "Dapple": 14018,
            },
        },
        "pp4 tp8 dp1 mb16":{
            "homo":{
                "OctoPipe": 599,
                "Interleaved": 565,
                "ZBV": 534,
                "ZBH": 553,
                "Dapple": 540,
            },
            "heter":{
                "OctoPipe": 8473,
                "Interleaved": 13298,
                "ZBV": 12072,
                "ZBH": 13172,
                "Dapple": 13018,
            },
        },
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "OctoPipe": 810,
                "Interleaved": 400,
                "ZBV": 775,
                "ZBH": 750,
                "Dapple": 720,
            },
            "heter":{
                "OctoPipe": 8964,
                "Interleaved": 0, #14534,
                "ZBV": 12632,
                "ZBH": 14232,
                "Dapple": 15134,
            },
        },
    },
    "42B LLaMA": {
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "OctoPipe": 1260,
                "Interleaved": 1207,
                "ZBV": 1040,
                "ZBH": 1180,
                "Dapple": 1128,
            },
            "heter":{
                "OctoPipe": 5528,
                "Interleaved": 9566,
                "ZBV": 9384,
                "ZBH": 9584,
                "Dapple": 10106,
            },
        },
        "pp4 tp4 dp2 mb16":{
            "homo":{
                "OctoPipe": 1375,
                "Interleaved": 1325,
                "ZBV": 1275,
                "ZBH": 1270,
                "Dapple": 1226,
            },
            "heter":{
                "OctoPipe": 5222,
                "Interleaved": 8462,
                "ZBV": 8440,
                "ZBH": 8240,
                "Dapple": 10506,
            },
        },
        "pp4 tp8 dp1 mb16":{
            "homo":{
                "OctoPipe": 955,
                "Interleaved": 920,
                "ZBV": 855,
                "ZBH": 890,
                "Dapple": 843,
            },
            "heter":{
                "OctoPipe": 5222,
                "Interleaved": 8202,
                "ZBV": 8340,
                "ZBH": 8240,
                "Dapple": 10306,
            },
        },
        "pp4 tp4 dp1 mb16":{
            "homo":{
                "OctoPipe": 1309,
                "Interleaved": 1257,
                "ZBV": 1271,
                "ZBH": 1240,
                "Dapple": 1195,
            },
            "heter":{
                "OctoPipe": 5525,
                "Interleaved": 8502,
                "ZBV": 8640,
                "ZBH": 8240,
                "Dapple": 10806,
            },
        },
        "pp8 tp2 dp1 mb32":{
            "homo":{
                "OctoPipe": 1395,
                "Interleaved": 0,
                "ZBV": 0,
                "ZBH": 0,
                "Dapple": 1334,
            },
            "heter":{
                "OctoPipe": 5528,
                "Interleaved": 0,
                "ZBV": 0,
                "ZBH": 0,
                "Dapple": 10156,
            },
        },
    },
    "14B LLaMA": {
        "pp4 tp2 dp1 mb16":{
            "homo":{
                "OctoPipe": 3776,
                "Interleaved": 3593,
                "ZBV": 3515,
                "ZBH": 3584,
                "Dapple": 3375,
            },
            "heter":{
                "OctoPipe": 2175,
                "Interleaved": 3506,
                "ZBV": 3456,
                "ZBH": 3456,
                "Dapple": 3650,
            },
        },
        "pp8 tp1 dp1 mb32":{
            "homo":{
                "OctoPipe": 3455,
                "Interleaved": 3262,
                "ZBV": 0,
                "ZBH": 3288,
                "Dapple": 3130,
            },
            "heter":{
                "OctoPipe": 2535,
                "Interleaved": 0, #4598,
                "ZBV": 0, #4536,
                "ZBH": 0, #4536,
                "Dapple": 4778,
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
                "Dapple": 670,
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
                "Dapple": 745,
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
                "Dapple": 3124,
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
                "OctoPipe": 1624,
                "Interleaved": 1760,
                "ZBV": 1663,
                "ZBH": 1712,
                "Dapple": 1904,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp1 mb16":{
            "heter":{
            },
            "homo":{
                "OctoPipe": 1624,
                "Interleaved": 1760,
                "ZBV": 1663,
                "ZBH": 1712,
                "Dapple": 1904,
            },
        },
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "OctoPipe": 1744,
                "Interleaved": 1864,
                "ZBV": 1751,
                "ZBH": 1808,
                "Dapple": 2032,
            },
        },
    },
    "70B LLaMA": {
        "pp4 tp4 dp4 mb16":{
            "homo":{
                "OctoPipe": 2039,
                "Interleaved": 2180,
                "ZBV": 2059,
                "ZBH": 2120,
                "Dapple": 2360,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp2 mb16":{
            "homo":{
                "OctoPipe": 2039,
                "Interleaved": 2180,
                "ZBV": 2059,
                "ZBH": 2120,
                "Dapple": 2360,
            },
            "heter":{
            },
        },
        "pp8 tp4 dp2 mb32":{
            "homo":{
                "OctoPipe": 2122,
                "Interleaved": 2290,
                "ZBV": 2149,
                "ZBH": 2220,
                "Dapple": 2500,
            },
            "heter":{
            },
        },
    },
    "14B LLaMA": {
        "pp8 tp1 dp1 mb32":{
            "homo":{
                "OctoPipe": 2332,
                "Interleaved": 2386,
                "ZBV": 2283,
                "ZBH": 2292,
                "Dapple": 2462,
            },
            "heter":{
            },
        },
        "pp4 tp2 dp1 mb16":{
            "homo":{
                "OctoPipe": 1698,
                "Interleaved": 1796,
                "ZBV": 1726,
                "ZBH": 1746,
                "Dapple": 1898,
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
                "OctoPipe": 6383,
                "Interleaved": 6836,
                "ZBV": 6454,
                "ZBH": 6738,
                "Dapple": 7370,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp1 mb16":{
            "homo":{
                "OctoPipe": 5802,
                "Interleaved": 6300,
                "ZBV": 6454,
                "ZBH": 6120,
                "Dapple": 6840,
            },
            "heter":{
            },
        },
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "OctoPipe": 2122,
                "Interleaved": 8332,
                "ZBV": 7031,
                "ZBH": 7314,
                "Dapple": 8078,
            },
            "heter":{
            },
        },
    },
    "42B LLaMA": {
        "pp4 tp4 dp1 mb16":{
            "homo":{
                "OctoPipe": 4002,
                "Interleaved": 4316,
                "ZBV": 4090,
                "ZBH": 4242,
                "Dapple": 4634,
            },
            "heter":{
            },
        },
        "pp4 tp4 dp2 mb16":{
            "homo":{
                "OctoPipe": 4002,
                "Interleaved": 4316,
                "ZBV": 4090,
                "ZBH": 4242,
                "Dapple": 4634,
            },
            "heter":{
            },
        },
        "pp4 tp8 dp1 mb16":{
            "homo":{
                "OctoPipe": 4002,
                "Interleaved": 4316,
                "ZBV": 4090,
                "ZBH": 4242,
                "Dapple": 4634,
            },
            "heter":{
            },
        },
        "pp8 tp4 dp1 mb32":{
            "homo":{
                "OctoPipe": 4554,
                "Interleaved": 4620,
                "ZBV": 4657,
                "ZBH": 4786,
                "Dapple": 5270,
            },
            "heter":{
            },
        },
    },
    "14B LLaMA": {
        "pp8 tp1 dp1 mb32":{
            "homo":{
                "OctoPipe": 2252,
                "Interleaved": 2386,
                "ZBV": 2283,
                "ZBH": 2292,
                "Dapple": 2462,
            },
            "heter":{
            },
        },
        "pp4 tp2 dp1 mb16":{
            "homo":{
                "OctoPipe": 1698,
                "Interleaved": 1796,
                "ZBV": 1726,
                "ZBH": 1746,
                "Dapple": 1898,
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
                "OctoPipe": [66.4321, 49.3999, 49.1665, 48.0542, 46.8794, 45.813, 44.7632, 68.9313 - 5], #TODO runtime wrong order
                "Interleaved": [66.103, 47.5874, 45.2378, 42.8882, 40.5386, 38.189, 35.8394, 52.5878],
                # "ZBV": ["OOM","OOM","OOM","OOM","OOM","OOM","OOM","OOM"],
                "ZBH": [56.7446, 41.4009, 41.1216, 40.8423, 40.563, 40.2837, 40.0044, 66.2174],
                "Dapple": [56.7446, 38.8638, 36.5767, 34.2896, 32.0025, 29.7153, 27.7252, 47.2957],
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
                "OctoPipe": [67.042, 49.749, 49.459, 48.35, 47.24, 46.421, 45.311, 62.975],
                "Interleaved": [65.296, 47.421, 45.202, 42.984, 40.765, 38.547, 36.328, 53.18],
                # "ZBV": [79.841, 44.093, 44.093, 44.093, 45.202, 45.202, 46.311, 46.311],
                "ZBH": [59.346, 43.732, 43.153, 42.573, 41.994, 41.415, 40.836, 67.698],
                "Dapple": [57.531, 39.656, 37.437, 35.219, 33.0, 30.781, 28.563, 45.852],
            },
            "heter":{
            },
        },
    },
}
vary_seq = {
    "OctoPipe": [1752, 2345, 2582, 2630, 2596],
    "ZBH": [1772, 2304, 2489, 2530, 2490],
    "Dapple": [1730, 2250, 2450, 2445, 2430],
}
