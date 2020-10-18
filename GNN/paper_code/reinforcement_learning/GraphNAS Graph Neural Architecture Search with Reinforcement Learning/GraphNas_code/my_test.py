class Father(object):

    def __init__(self):
        print("调用了Father的构造函数")
        self.get_name()

    def get_name(self):
        print("Father类中get_name函数")


class Child(Father):

    def __init__(self):
        print("ok")
        super(Child, self).__init__()

    def get_name(self):
        print("Child类中的get_name函数")


if __name__ == "__main__":
    A = Child()
    i = 1
    print("1",i,"1")