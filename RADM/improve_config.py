
class Config:
    TASI_ENABLED = False
    SERM = False
    DUAL_STREAM_SERM = False
    DUAL_GTRAM = False
    SERM_K = 5

    # 禁止运行时覆盖
    def __setattr__(self, key, value):
        raise RuntimeError("Config is read-only! Clone or use env-var override.")
    

# 把类当单例用，不实例化
config = Config()
