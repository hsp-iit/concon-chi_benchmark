# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import logging
import re
from rich.traceback import install
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn,\
                          MofNCompleteColumn, TimeRemainingColumn, TaskProgressColumn

    
class CustomFormatter(logging.Formatter):
    def __init__(self, to_file=False) -> None:
        super().__init__()
        self.to_file = to_file

    format = f" - %(message)s [grey54](%(pathname)s:%(lineno)d - %(asctime)s)[/]"

    FORMATS = {
        logging.DEBUG: f"[grey]%(levelname)s[/]" + format,
        logging.INFO: f"[blue]%(levelname)s[/]" + format,
        logging.WARNING: f"[yellow]%(levelname)s[/]" + format,
        logging.ERROR: f"[red]%(levelname)s[/]" + format,
        logging.CRITICAL: f"[b blink red]%(levelname)s[/]" + format,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        out = formatter.format(record)
        return out if not self.to_file  else re.sub("\[.*?\]","",out)
    
progress_bars = Progress(
    SpinnerColumn(finished_text="🦝 "), TextColumn("[progress.description]{task.description}"),
    BarColumn(), MofNCompleteColumn(), TaskProgressColumn(),
    TextColumn('('), TimeRemainingColumn(), TimeElapsedColumn(), TextColumn(')')
)

LEVEL = 9 

file_handler = None
rich_handler = RichHandler(show_path=False, show_level=False, show_time=False, markup=True)
rich_handler.setFormatter(CustomFormatter())

logger = logging.getLogger('rich')
logger.propagate = False
logger.setLevel(level=LEVEL)
logger.addHandler(rich_handler)
    
def enable_file_handler(file_path):
    global file_handler
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(CustomFormatter(to_file=True))
    logger.addHandler(file_handler)
    
logging.addLevelName(LEVEL, "PROGRESS")
def progress(self, task_id=None, description=None, total=None, advance=1, visible=None, stop=False):
    if not self.isEnabledFor(LEVEL):
        return
    
    if not progress_bars.live.is_started:
        progress_bars.start()
    
    if stop:     
        for t in progress_bars.tasks:
            if t.completed < t.total and file_handler is not None:
                msg = f'{t.description} {t.completed}/{t.total}'
                record = logging.LogRecord('rich', LEVEL, pathname=None, lineno=None, msg=msg, args=None, exc_info=None, func=None, sinfo=None)
                file_handler.emit(record)
        
        progress_bars.stop()
        return
    
    if task_id is None:
        return progress_bars.add_task(description=description, total=total, start=False)
    
    task = progress_bars.tasks[task_id]
    
    if task.completed == 0:
        progress_bars.start_task(task_id)
    
    progress_bars.update(task_id, advance=advance, visible=visible)
    
    if task.completed >= task.total and file_handler is not None:
        msg = f'{task.description} {task.completed}/{task.total} Elapsed: {task.finished_time:.2f}s'
        record = logging.LogRecord('rich', LEVEL, pathname=None, lineno=None, msg=msg, args=None, exc_info=None, func=None, sinfo=None)
        file_handler.emit(record)

install(show_locals=False, width=250)

logging.Logger.progress = progress




