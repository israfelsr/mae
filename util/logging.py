import logging

def get_logger(name, filename=None, level=logging.DEBUG):
    """Returns a logger that logs to stdout and a filename if specified.
    Args:
        name: name of the module
        filename: (optional) name of the output file
        level: logging level to capture.
    Returns:
        The singleton Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger