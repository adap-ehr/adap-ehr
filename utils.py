import os.path as osp
import logging


class Logger:
    def __init__(self, save_dir):
        if save_dir is not None:
            self.logger = logging.getLogger()
            logging.basicConfig(filename = osp.join(save_dir,"experiment.log"),format='%(asctime)s | %(message)s')
            logging.root.setLevel(level=logging.INFO)
        else:
            self.logger = None

    def info(self, msg, to_file=True):
        print(msg)
        if self.logger is not None and to_file:
            self.logger.info(msg)

    def close(self):
        self.logger.handlers[0].stream.close()
        self.logger.removeHandler(self.logger.handlers[0])
