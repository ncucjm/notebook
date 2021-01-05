from _test_class_ import EV
import ray
ray.shutdown()
#ray　初始化
ray.init()

task = EV.remote(10)
c = task.run_ea.remote()
a = ray.get(c)
b = task.run_a.remote(a)
d = ray.get(b)
print(a)
print(d)


# @ray.remote
# class Foo(object):
#
#     # Any method of the actor can return multiple object refs.
#     @ray.method(num_returns=3)
#     def bar(self):
#         return 1, 2, 3
#
# f = Foo.remote()
#
# obj_ref1, obj_ref2, obj_ref3 = f.bar.remote()
# print(ray.get(obj_ref1))
# print(ray.get(obj_ref2))
# print(ray.get(obj_ref3))


