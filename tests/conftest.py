"""Keep each test run's temporary files inside this isolated project."""
import uuid
from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        directory=Path(config.rootpath)/'.cache'/'pytest'
        directory.mkdir(parents=True,exist_ok=True)
        run=directory/uuid.uuid4().hex
        assert run.resolve().is_relative_to(directory.resolve()) and not run.exists()
        config.option.basetemp=str(run)
