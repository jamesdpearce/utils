import sys
import time

class ProgressBar(object):
    def __init__(self, bar_length=20):
        self.start = time.time()
        self.bar_length = bar_length

    def reset_time(self):
        self.start = time.time()

    def update_progress(self, progress):
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        block = int(round(self.bar_length * progress))
        elapsed_time = time.time() - self.start
        if progress > 0:
            estimated_time_remaining = ((1 - progress) / (progress)) * elapsed_time
            status = '    Elapsed: %is     Remaining: %is' % (elapsed_time, estimated_time_remaining)
        if progress >= 1:
            progress = 1
            status = "    Elapsed: %is\r\n" % elapsed_time

        text = "\rPercent: [%s] %.2f%% %s" % ("#"*block + "-"*(self.bar_length - block),
                                              progress*100,
                                              status)
        sys.stdout.write(text)
        sys.stdout.flush()


class ProgressCounter(object):
    def __init__(self):
        self.start = time.time()
        self.progress = 0

    def reset_time(self):
        self.start = time.time()

    def increment_progress(self, increment=1):
        status = ""
        if not isinstance(increment, int):
            increment = 0
            status = "error: progress increment var must be float\r\n"
        self.progress += increment

        elapsed_time = time.time() - self.start
        rate = self.progress / elapsed_time
        status = '\rProgress: %i     Elapsed: %is     Rate: %.3f / sec' % (self.progress, elapsed_time, rate)
        sys.stdout.write(status)
        sys.stdout.flush()
