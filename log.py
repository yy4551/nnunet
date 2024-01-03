import logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler(r"C:\Git\NeuralNetwork\nnunet\log.txt",mode='w')
formatter = logging.Formatter('%(module)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(level = logging.DEBUG)