from simulator.ReCycleSimulator import ReCyclePipeline

if __name__ == "__main__":
    pipeline_num = 4
    stage_num = 4
    device_num = stage_num
    microbatch_num = [stage_num * 3 for _ in range(pipeline_num)]
    comp_time = 20
    comp_time_ratio = [[1 for _ in range(stage_num)] for _ in range(pipeline_num)]
    comp_time_ratio[0][2] = 2

    # pipeline_num = 4
    # stage_num = 4
    # device_num = stage_num
    # microbatch_num = [stage_num * 4 for _ in range(pipeline_num)]
    # microbatch_num = [16,12,16,20]
    # comp_time = 20
    # comp_time_ratio = [
    #     [1,1,1,1],
    #     [1,2,1,1],
    #     [1,1,1,1],
    #     [1,1,1,1],
    # ]
    assert len(comp_time_ratio) == pipeline_num
    assert len(comp_time_ratio[0]) == stage_num
    config = {
        'pipeline_num': pipeline_num,
        'device_num': stage_num,
        'microbatch_num': microbatch_num,
        'stage_num': stage_num,
        'fail_pipelines_stages': {0:[0], 1:[0, 1], 2:[2], 3:[3]},
        'file_path': "recycle_test_res.txt",
        'time_limit': 30,
        'f_time': [comp_time * 1.0 for _ in range(stage_num)],
        'b_time': [comp_time * 1.5 for _ in range(stage_num)],
        'w_time': [comp_time * 0.5 for _ in range(stage_num)],
        'comp_time_ratio': comp_time_ratio,
    }
    sim = ReCyclePipeline(
        config=config,
        new_comm_length=1,
    )
    sim.run(draw=True)