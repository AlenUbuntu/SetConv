import os 
import logging 
import sys



def setup_logger(name, save_dir=None, filename='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # stream handler
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # file handler
    if save_dir:
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger 
    