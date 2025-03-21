labels = {
    "UnifiedPP": "UnifiedPP",
    "Interleaved-L": "Interleaved-L",
    "Interleaved": "Interleaved",
    "ZBV": "ZBV",
    "ZBH": "ZBH",
    "1F1B": "1F1B",
}

data = {
    "56B Llama": {
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
    "70B Llama": {
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

colors = {
    "UnifiedPP": "#F8CECC",
    "Interleaved-L": "#DAE8FC",
    "Interleaved": "#99CCFF",
    "ZBV": "#D5E8D4",
    "ZBH": "#FFE6CC",
    "1F1B": "#FFF2CC",
}

sim = {
    "56B Llama": {
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
    "70B Llama": {
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
}

mem = {
    "56B Llama": {
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
    "70B Llama": {
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
}

mem_sim = {
    "56B Llama": {
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
    "70B Llama": {
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
}
