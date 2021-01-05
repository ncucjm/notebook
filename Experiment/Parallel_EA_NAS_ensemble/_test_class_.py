import ray
# ray.shutdown()
# #ray　初始化
# ray.init()
@ray.remote
class EV(object):

    #@ray.method(num_returns=2)
    def __init__(self, y):
        self.a = y

    @ray.method(num_returns=1)
    def run_ea(self):
        return self.a
    @ray.method(num_returns=1)
    def run_a(self, x):
        return x + 1