import logging

def setup_logger(filename):

    # Add custom logging level #justvirgothings
    IMPORTANT_LEVEL_NUM = 25
    logging.addLevelName(IMPORTANT_LEVEL_NUM, "IMPORTANT")
    def important(self, message, *args, **kws):
        # Yes, logger takes its '*args' as 'args'.
        self._log(IMPORTANT_LEVEL_NUM, message, args, **kws)
    logging.Logger.important = important

    logger = logging.getLogger('log-eval')
    logger.setLevel(logging.DEBUG)

    # Create file handler
    log_file_handler = logging.FileHandler(filename, 'w+')
    log_file_handler.setLevel(logging.DEBUG)

    # Create console handler
    log_console_handler = logging.StreamHandler()
    log_console_handler.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                  "%Y-%m-%d %H:%M:%S")

    log_file_handler.setFormatter(formatter)
    log_console_handler.setFormatter(formatter)

    logger.addHandler(log_file_handler)
    logger.addHandler(log_console_handler)

    return logger