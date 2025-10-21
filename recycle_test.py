from simulator.ReCycleSimulator import ReCyclePipeline

if __name__ == "__main__":
    pipeline_num = 4
    stage_num = 4
    device_num = stage_num
    microbatch_num = [stage_num * 4 for _ in range(pipeline_num)]
    microbatch_num = [16,12,16,20]
    comp_time = 20
    config = {
        'pipeline_num': 4,
        'device_num': stage_num,
        'microbatch_num': microbatch_num,
        'stage_num': stage_num,
        'fail_pipelines_stages': {1:[2]},
        'file_path': "recycle_test_res.txt",
        'time_limit': 3,
        'f_time': [comp_time for _ in range(stage_num)],
        'b_time': [comp_time for _ in range(stage_num)],
        'w_time': [comp_time for _ in range(stage_num)],
    }
    sim = ReCyclePipeline(
        config=config,
        new_comm_length=1,
    )
    sim.run(draw=True)