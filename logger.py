class Log:
    def __init__(self, log_interval=50):
        self.log_interval = log_interval
        self.loss = 0
        self.precision = 0

    def __call__(self, msg, *args):
        print(msg.format(args))

    def __call__(self, batch_id, msg, *args):
        if batch_id % self.log_interval == 0:
            print(msg.format(args))
