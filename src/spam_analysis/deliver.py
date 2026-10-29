"""Generate final figures, verified course notebooks and report from saved runs."""
from .result_plots import build as plots
from .notebooks import build as notebooks,execute
from .reporting import build_report
from .reproducibility import environment_snapshot,verify_predictions


def deliver(config):
    verify_predictions(); environment_snapshot(); plots(); notebooks(); execute(); build_report()
    print('Report, figures and executed notebooks generated from frozen saved results.',flush=True)
