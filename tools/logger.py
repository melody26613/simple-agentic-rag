import logging
import logging.handlers
import os
import sys

LOG_FORAMT = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGDIR = "log"

handlers = {}
handled_loggers = []


def build_logger(logger_name, logger_filename, log_dir=LOGDIR):
    global handlers, handled_loggers

    formatter = LOG_FORAMT
    filename = os.path.join(log_dir, logger_filename)
    handler = handlers.get(filename, None)

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when="D", utc=True, encoding="UTF-8"
        )
        handler.setFormatter(formatter)
        handlers[filename] = handler

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger) and name not in handled_loggers:
                item.addHandler(handler)
    else:
        logger.addHandler(handler)

    handled_loggers.append(logger_name)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""
