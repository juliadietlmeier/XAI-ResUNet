import logging
import os

def setup_logging(model_name):
    log_dir = f'models/{model_name}/logs'
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('xairesunet_logger')
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to avoid duplicate logs
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
