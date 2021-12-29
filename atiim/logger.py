import logging
import sys


class Logger:
    """Initialize project-wide logger. The logger outputs to both stdout and a file."""

    # output format for log string
    LOG_FORMAT_STRING = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @property
    def log_format(self):
        """Generate log formatter."""

        return logging.Formatter(self.LOG_FORMAT_STRING)

    @property
    def logger(self):
        """Initialize logger as level info."""

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        return logger

    def initialize_logger(self):
        """Initialize logger to stdout and file."""

        # logger console handler
        self.console_handler()

    def console_handler(self, log_level):
        """Construct console handler."""

        console_handler = logging.StreamHandler(sys.stdout)

        if log_level == 'debug':
            console_handler.setLevel(logging.DEBUG)
        elif log_level == 'info':
            console_handler.setLevel(logging.INFO)
        elif log_level == 'warning':
            console_handler.setLevel(logging.WARNING)
        else:
            console_handler.setLevel(logging.ERROR)

        console_handler.setFormatter(self.log_format)
        self.logger.addHandler(console_handler)

    @staticmethod
    def close_logger():
        """Shutdown logger."""

        # Remove logging handlers
        logger = logging.getLogger()

        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        logging.shutdown()
