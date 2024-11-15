from abstract.Pipeline import PipelineScheduler

if __name__ == "__main__":
    pp = PipelineScheduler()
    pp.run_pipeline_parallelism(time_limit=2000)
    pp.show_mem_usage()
    pp.draw()
    pass

